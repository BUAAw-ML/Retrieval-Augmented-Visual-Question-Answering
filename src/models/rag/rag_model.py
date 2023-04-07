

import copy
import math
import os
from turtle import forward
import warnings

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from torch.utils.checkpoint import checkpoint
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union
from collections import Counter, defaultdict
from easydict import EasyDict
from transformers import T5Tokenizer, T5Config,T5ForConditionalGeneration, T5PreTrainedModel #
from transformers import VisualBertModel, VisualBertConfig, BertTokenizer
from transformers import DPRQuestionEncoder, DPRContextEncoder, DPRConfig
from transformers import BertModel, BertConfig
from transformers.models.rag.retrieval_rag import CustomHFIndex, CanonicalHFIndex
import pytorch_lightning as pl
import numpy as np
import torch.nn.functional as F

import time
import sys


class RagModel(pl.LightningModule):
    '''
    Class for RAG, re-implementation
    '''
    def __init__(self, config: EasyDict, data_loader) -> None:
        super().__init__()

        self.config = config
        self.data_loader = data_loader
        self.retriever_tokenizer = data_loader.tokenizer
        self.generator_tokenizer = data_loader.decoder_tokenizer

        
        # Initialising question encoder
        QueryEncoderModelClass = globals()[self.config.model_config.QueryEncoderModelClass]
        QueryEncoderConfigClass = globals()[self.config.model_config.QueryEncoderConfigClass]
        question_encoder_model_config = QueryEncoderConfigClass.from_pretrained(self.config.model_config.QueryEncoderModelVersion, local_files_only=True, cache_dir="../public") #, cache_dir="../public"
        self.question_encoder = QueryEncoderModelClass.from_pretrained(self.config.model_config.QueryEncoderModelVersion,local_files_only=True,  cache_dir="../public",
                                                    config=question_encoder_model_config)
        self.retiever_hidden_size = question_encoder_model_config.hidden_size

        # Initialising generator
        GeneratorModelClass = globals()[self.config.model_config.GeneratorModelClass]
        GeneratorConfigClass = globals()[self.config.model_config.GeneratorConfigClass]
        generator_model_config = GeneratorConfigClass.from_pretrained(self.config.model_config.GeneratorModelVersion,local_files_only=True,  cache_dir="../public")
        self.generator = GeneratorModelClass.from_pretrained(self.config.model_config.GeneratorModelVersion, local_files_only=True, cache_dir="../public",
                                                    config=generator_model_config)
        
        self.question_encoder.resize_token_embeddings(len(self.retriever_tokenizer))
        self.generator.resize_token_embeddings(len(self.generator_tokenizer))
        
        self.loss_fct = CrossEntropyLoss(ignore_index=-100)

        if 'add_null_document' in self.config.model_config.modules:
            # Add an embedding for null document!
            self.null_embedding = nn.Parameter(torch.zeros(self.retiever_hidden_size))

        if 'read_static_retrieval_results' in self.config.model_config.modules:
            self.retrieve = self.static_retrieve
        else:
            self.retrieve = self.main_retrieve

        self.num_attention_heads = 12
        self.hidden_size = 1024
        self.attention_head_size = int(self.hidden_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(self.hidden_size, self.all_head_size)
        self.key = nn.Linear(self.hidden_size, self.all_head_size)
        self.value = nn.Linear(self.hidden_size, self.all_head_size)
    

    def init_retrieval(self):

        if 'read_static_retrieval_results' in self.config.model_config.modules:
            # Load static retrieval results
            self.questionId2topPassages = self.data_loader.data.vqa_data_with_dpr_output.questionId2topPassages.copy()
            return
   
        if self.config.data_loader.index_files.index_passages_path == '':
            # use wikidata
            self.index = CanonicalHFIndex(
                vector_size=self.retiever_hidden_size,
                dataset_name=self.config.data_loader.index_files.index_dataset,
                dataset_split=self.config.data_loader.index_files.index_dataset_split,
                index_name=self.config.data_loader.index_files.index_name,
                index_path=None,
                use_dummy_dataset=True if self.config.data_loader.index_files.index_dummy else False,
            )
            self.data_source = 'wiki'
        else:

            # use GS corpus
            self.index = CustomHFIndex.load_from_disk(
                vector_size=self.retiever_hidden_size,
                dataset_path=self.config.data_loader.index_files.index_passages_path,
                index_path=self.config.data_loader.index_files.index_path,
            )

            self.data_source = 'gs'
        print("initializing retrieval")
        self.index.init_index()
        # print(self.index)
        # input('init finished!')

    def main_retrieve(self, 
                    input_ids: torch.Tensor,
                    attention_mask: torch.Tensor, 
                    labels: torch.Tensor, 
                    question_ids: List, 
                    input_text_sequences: List, 
                    n_docs=None,
                    **kwargs):
        """ Main retrieval function, retrieve documents using retriever

        Args:
            input_ids (torch.Tensor): [description]
            attention_mask (torch.Tensor): [description]
            labels (torch.Tensor): [description]
            question_ids (List): [description]
            input_text_sequences (List): [description]
            n_docs ([type], optional): [description]. Defaults to None.

        Returns:
            [type]: [description]
        """
        if n_docs is None:
            n_docs = self.config.data_loader.additional.num_knowledge_passages

        batch_size = input_ids.shape[0]

        # Use question_encoder to encode question inputs
        query_outputs = self.question_encoder(input_ids=input_ids,
                                            attention_mask=attention_mask)
        question_hidden_states = query_outputs.pooler_output
        # print('question_hidden_states', question_hidden_states.shape)

        

        start_time = time.time()
        ids, vectors = self.index.get_top_docs(question_hidden_states.cpu().detach().numpy(), n_docs)

        # print(
        #     f"index search time: {time.time() - start_time} sec, batch size {question_hidden_states.shape}"
        # )
        # print(ids)

        # question_hidden_states: batch_size x hidden_size
        # item_hidden_states: batch_size x n_docs x hidden_size
        item_hidden_states = torch.Tensor(vectors).type_as(question_hidden_states)

        # print('item_hidden_states', item_hidden_states.shape)

        doc_scores = (question_hidden_states.unsqueeze(dim=1) * item_hidden_states).sum(dim=-1)
        
        
        if 'add_null_document' in self.config.model_config.modules:
            null_doc_scores = (question_hidden_states * self.null_embedding.unsqueeze(dim=0)).sum(dim=-1)
            # null_doc_scores: batch_size
            # print('null_doc_scores', null_doc_scores)
            
        doc_scores_cpu = doc_scores.cpu().detach().numpy()
        
        retrieved_docs = []
        for b in range(batch_size):
            doc_data = []
            contents = self.index.get_doc_dicts(ids[b])
            if 'add_null_document' in self.config.model_config.modules:
                passage_data = {
                    'passage_id': "0",
                    'content': "",
                    'score': null_doc_scores.cpu().detach().numpy()[b],
                }
                doc_data.append(passage_data)

            for i in range(n_docs):
                if self.data_source == 'wiki':
                    content = 'title: ' + contents[i]['title'] + " content: " + contents[i]['text']
                else:
                    content = contents[i]['text']
                content = ' '.join(['<BOK>', content, '<EOK>'])
                passage_data = {
                    'passage_id': str(ids[b, i]),
                    'content': content,
                    'score': doc_scores_cpu[b, i]
                }
                # print(content)
                # print(self.data_loader.data.passages.id2doc[str(ids[b, i])])
                # input()
                doc_data.append(passage_data)
            retrieved_docs.append(doc_data)
        
        assert len(retrieved_docs) == batch_size

        # print(retrieved_docs)
        if 'add_null_document' in self.config.model_config.modules:
            # print('doc_scores', doc_scores.shape) # batch_size x n_docs
            doc_scores = torch.cat([
                null_doc_scores.reshape(batch_size, 1), # batch_size x 1
                doc_scores,
            ], dim=-1)
            # print('after doc_scores', doc_scores.shape) # batch_size x n_docs
            # input()

        return EasyDict(
            retrieved_docs=retrieved_docs,
            doc_scores=doc_scores,
            question_hidden_states=question_hidden_states,
        )

    def static_retrieve(self, 
                    input_ids: torch.Tensor,
                    attention_mask: torch.Tensor, 
                    labels: torch.Tensor, 
                    question_ids: List, 
                    input_text_sequences: List, 
                    n_docs=None,
                    **kwargs):
        """A dummy retrieval function, retrieve from static results

        Args:
            input_ids (torch.Tensor): [description]
            attention_mask (torch.Tensor): [description]
            labels (torch.Tensor): [description]
            question_ids (List): [description]
            input_text_sequences (List): [description]
            n_docs ([type], optional): [description]. Defaults to None.

        Returns:
            [type]: [description]
        """
        if n_docs is None:
            n_docs = self.config.data_loader.additional.num_knowledge_passages
        
        batch_size = input_ids.shape[0]

        #####   Dummy Retrieval ####
        retrieved_docs = []
        doc_scores = []
        for question_id in question_ids:
            annotation = self.questionId2topPassages[str(question_id)]
            top_passages = annotation[:n_docs]
            retrieved_docs.append(top_passages)
            scores = [p['score'] for p in top_passages]
            doc_scores.append(scores)
        
        doc_scores = torch.FloatTensor(doc_scores).type_as(input_ids)

        assert len(retrieved_docs) == batch_size

        return EasyDict(
            retrieved_docs=retrieved_docs,
            doc_scores=doc_scores,
        )

    def prepare_inputs_for_generator(self, 
                input_text_sequences, retrieved_docs, labels, n_docs=None):
        
        if n_docs is None:
            n_docs = self.config.data_loader.additional.num_knowledge_passages
        
        batch_size = len(input_text_sequences)

        extended_input_text_sequences = []

        for index, input_text_sequence in enumerate(input_text_sequences):
            scores = []
            for doc in retrieved_docs[index]:#[:-1]:
                extended_input_text_sequences.append(
                    ' '.join([input_text_sequence, doc['content']])
                )
                scores.append(doc['score'])
            # extended_input_text_sequences.append(input_text_sequence)
            # scores.append(0)
        

        targets = labels

        encoding = self.generator_tokenizer([sequence for sequence in extended_input_text_sequences],
                                    padding='longest',
                                    max_length=self.config.data_loader.additional.max_decoder_source_length,
                                    truncation=True,
                                    return_tensors="pt")

        generator_input_ids, generator_attention_mask = encoding.input_ids, encoding.attention_mask
        generator_input_ids = generator_input_ids.to(labels.device)

        generator_attention_mask = generator_attention_mask.to(labels.device)
        generator_decoder_input_ids = self.generator._shift_right(targets)

        encoding_embeddings = self.generator.get_input_embeddings()(encoding.input_ids.to(labels.device))

        return EasyDict(
            generator_input_text_sequences=extended_input_text_sequences,
            generator_input_ids=generator_input_ids,
            generator_input_embeddings=encoding_embeddings,
            generator_attention_mask=generator_attention_mask,
            generator_decoder_input_ids=generator_decoder_input_ids,
            generator_labels=targets,
        )

    def prepare_inputs_for_generator_by_preprocessDoc(self, 
                input_text_sequences, retrieved_docs, labels, n_docs=None):
    

        if n_docs is None:
            n_docs = self.config.data_loader.additional.num_knowledge_passages
        
        batch_size = len(input_text_sequences)
        
        extended_input_text_sequences = []

        for index, input_text_sequence in enumerate(input_text_sequences):
            scores = []
            for doc in retrieved_docs[index]:
                extended_input_text_sequences.append(
                    ' '.join([input_text_sequence, doc['content']])
                )
                scores.append(doc['score'])
        
        def transpose_for_scores(x):
            new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
            x = x.view(*new_x_shape)
            return x.permute(0, 2, 1, 3)

        encoding = self.generator_tokenizer([sequence for sequence in extended_input_text_sequences],
            padding='longest',
            max_length=self.config.data_loader.additional.max_decoder_source_length, # - len(input_text_sequences[0])
            truncation=True,
            return_tensors="pt")

        knowledge_mask = encoding.attention_mask
        input_text_ids = self.generator_tokenizer.convert_tokens_to_ids(self.generator_tokenizer.tokenize(input_text_sequences[0]))
        knowledge_mask[:,:len(input_text_ids)] = 0
        knowledge_mask = knowledge_mask.to(labels.device).unsqueeze(-2).unsqueeze(-2)

        input_text_ids = torch.tensor(np.array(input_text_ids))
        input_text_embeddings = self.generator.get_input_embeddings()(input_text_ids.to(labels.device))
        input_text_embedding = torch.mean(input_text_embeddings,-2)
        encoding_embeddings = self.generator.get_input_embeddings()(encoding.input_ids.to(labels.device)) #[docs_num, L, 1024]

        query_layer = transpose_for_scores(self.query(input_text_embedding.unsqueeze(0).unsqueeze(0))) #[1,12,1,]
        key_layer = transpose_for_scores(self.key(encoding_embeddings))#[5,12,L,]
        value_layer = encoding_embeddings#[5,L,1024]

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores2 = attention_scores.masked_fill((1 - knowledge_mask.byte()).bool(), torch.tensor(-np.inf).to(labels.device)) #[5,12,1,L]
        
        attention_probs = nn.Softmax(dim=-1)(attention_scores2) #[5,12,1,L]

        knowledge_mask = knowledge_mask.sum(-1)
        knowledge_mask = knowledge_mask.unsqueeze(-2)
        attention_probs = knowledge_mask * attention_probs
        attention_probs2 = attention_probs.clone()

        attention_probs2[:,:,:,:len(input_text_ids)] = 1
        attention_probs2 = attention_probs2.mean(-3).transpose(-1, -2)  #[5,L,1]

 

        # print(attention_probs2.sum(-2))
        
        output_embeddings = attention_probs2 * value_layer 

        # extended_input_text_sequences = []
        # extended_input_text_sequences.extend(input_text_sequences)

        # for index, _ in enumerate(input_text_sequences):
        #     for doc in retrieved_docs[index]:
        #         extended_input_text_sequences.extend([doc['content']])
        # print(extended_input_text_sequences)
       # Get encoder outputs first
        # test_batch = EasyDict({
        #     'input_ids': encoding.input_ids.to(labels.device),
        #     'attention_mask': encoding.attention_mask.to(labels.device),
        #     'return_dict': True,
        # })
        # print(encoding.input_ids)
        # print(encoding.attention_mask)
        # encoder_outputs = self.generator.encoder(
        #     **test_batch
        # )

        # input_hidden_state = encoder_outputs.last_hidden_state[0] #shape[L,1024]
        # docs_hidden_state = encoder_outputs.last_hidden_state[1:] #

        # print(input_hidden_state.shape)
        # print(torch.mean(input_hidden_state,0).shape)

        # # shape[5,250,1024] shape[1024]
        # # attention = torch.matmul(docs_hidden_state, torch.mean(input_hidden_state,0))
        # masks = encoding.attention_mask[1:,:].to(labels.device)
        # print(masks.shape)
        # print(torch.matmul(docs_hidden_state, torch.mean(input_hidden_state,0)).shape)
        # attention = (torch.matmul(docs_hidden_state, torch.mean(input_hidden_state,0))).masked_fill(1 - masks.byte(), torch.tensor(-np.inf).to(labels.device))
        # attention = F.softmax(attention, -1)
        # print(attention)
        # print(attention.shape)
        # # print(encoder_outputs)
        # exit()

        targets = labels

        generator_input_ids, generator_attention_mask = encoding.input_ids, encoding.attention_mask
        generator_input_ids = generator_input_ids.to(labels.device)
        generator_attention_mask = generator_attention_mask.to(labels.device)
        generator_decoder_input_ids = self.generator._shift_right(targets)

        return EasyDict(
            generator_input_text_sequences=extended_input_text_sequences,
            generator_input_ids=generator_input_ids,
            generator_input_embeddings=output_embeddings,
            generator_attention_mask=attention_probs2.squeeze(-1),#generator_attention_mask,
            generator_decoder_input_ids=generator_decoder_input_ids,
            generator_labels=targets,
        )


    def forward(self, input_ids: torch.Tensor,
                      attention_mask: torch.Tensor,
                      labels: torch.Tensor,
                      question_ids: List,
                      input_text_sequences: List,
                    **kwargs):
        
        batch_size = input_ids.shape[0]

        # Retrieve docs for given question inputs
        retrieval_results = self.retrieve(input_ids, attention_mask, labels, question_ids, input_text_sequences)
        retrieved_docs, doc_scores = retrieval_results.retrieved_docs, retrieval_results.doc_scores
        
        answers = kwargs.get('answers', None)
        assert answers is not None
        get_retrieval_labels_results = self.get_retrieval_labels(
            batch_answers=answers,
            batch_retrieved_docs=retrieved_docs,
        )
        retrieval_labels = get_retrieval_labels_results.retrieval_labels


        n_docs = self.config.data_loader.additional.num_knowledge_passages
        if 'add_null_document' in self.config.model_config.modules:
            n_docs += 1
        
        if 'force_existence' in self.config.model_config.modules:
            # Force the label to be in the retrieved document
            selected_answers = get_retrieval_labels_results.selected_answers
            target_encoding = self.generator_tokenizer(selected_answers,
                    padding='longest',
                    max_length=self.config.data_loader.additional.max_target_length,
                    truncation=True)
            labels = target_encoding.input_ids
            labels = torch.LongTensor(labels).type_as(input_ids)
        else:
            labels = labels.repeat_interleave(n_docs, 0)


        # prepare inputs for generator
        generator_inputs = self.prepare_inputs_for_generator(input_text_sequences=input_text_sequences,
                                            retrieved_docs=retrieved_docs,
                                            labels=labels, n_docs=n_docs)
        

        # generator_inputs = self.prepare_inputs_for_generator_by_preprocessDoc(input_text_sequences=input_text_sequences,
        #                             retrieved_docs=retrieved_docs,
        #                             labels=labels, n_docs=n_docs)
        
        generator_outputs = self.generator(
                            input_ids=generator_inputs.generator_input_ids,
                            # inputs_embeds=generator_inputs.generator_input_embeddings,
                            attention_mask=generator_inputs.generator_attention_mask,
                            decoder_input_ids=generator_inputs.generator_decoder_input_ids,
                            return_dict=True)


        # #################################
        # # Get encoder outputs first
        # test_batch = EasyDict({
        #     'input_ids': generator_inputs.generator_input_ids,
        #     'attention_mask': generator_inputs.generator_attention_mask,
        #     'return_dict': True,
        # })

        # encoder_outputs = self.generator.encoder(
        #     **test_batch
        # )
        # print(input_text_sequences)
        # print(generator_inputs.generator_input_ids)

        # text_end = torch.nonzero(generator_inputs.generator_input_ids==32102)
        # print(text_end)
        
        # encoder_outputs['last_hidden_state'] = encoder_outputs[0][:,:text_end[0][1],:] #text_end[0][1]+1

        # # input_text_sequences.shape
        # generator_outputs = self.generator(
        #                     encoder_outputs=encoder_outputs, # use pre-computed encoder outputs
        #                     decoder_input_ids=generator_inputs.generator_decoder_input_ids,
        #                     return_dict=True)
        # #######################################

        logits = generator_outputs.logits

        loss_dict = self.get_loss(
            seq_logits=logits,
            doc_scores=doc_scores,
            target=generator_inputs.generator_labels,
            exclude_bos_score=False,
            n_docs=n_docs,
            retrieval_labels=retrieval_labels,
        )
        

        # aggregate loss
        total_loss = 0
        for loss_name, loss_ratio in self.config.model_config.loss_ratio.items():
            if loss_ratio != 0:
                total_loss += loss_dict[loss_name] * loss_ratio
        
        return EasyDict(loss=total_loss,
                        loss_dict=loss_dict,
                        doc_scores=doc_scores.cpu().detach().numpy())

        
        

    def generate(self, input_ids: torch.Tensor,
                      attention_mask: torch.Tensor,
                      labels: torch.Tensor,
                      question_ids: List,
                      input_text_sequences: List,
                      n_docs: int=None,
                      **kwargs):

        batch_size = input_ids.shape[0]
        
        # Retrieve docs for given question inputs
        retrieval_results = self.retrieve(input_ids, attention_mask, labels, question_ids, input_text_sequences)
        retrieved_docs, doc_scores = retrieval_results.retrieved_docs, retrieval_results.doc_scores
        # print(doc_scores)
        # doc_scores_mean = torch.mean(doc_scores,-1).unsqueeze(-1)
        # print(doc_scores_mean)
        # print(doc_scores-doc_scores_mean)




        if n_docs is None:
            n_docs = self.config.data_loader.additional.num_knowledge_passages
            if 'add_null_document' in self.config.model_config.modules:
                n_docs += 1

        # populate labels
        labels = labels.repeat_interleave(n_docs, 0)

        # prepare inputs for generator
        generator_inputs = self.prepare_inputs_for_generator(input_text_sequences=input_text_sequences,
                                            retrieved_docs=retrieved_docs,
                                            labels=labels,
                                            n_docs=n_docs)
        
        
        # Get encoder outputs first
        test_batch = EasyDict({
            'input_ids': generator_inputs.generator_input_ids,
            'attention_mask': generator_inputs.generator_attention_mask,
            'return_dict': True,
        })

        encoder_outputs = self.generator.encoder(
            **test_batch
        )

        # Get decoder outputs from encoder_outputs
        test_batch = {
            'encoder_outputs': encoder_outputs,
            "max_length": self.config.data_loader.additional.max_target_length,
        }
        generation_outputs = self.generator.generate(**test_batch)

        # generation_outputs_decoded = self.generator_tokenizer.batch_decode(generation_outputs, skip_special_tokens=True)
        # print(generation_outputs_decoded)
        # exit()
        

        # Find answer proposals from n_docs outputs for each question
        outputs = []
        generation_outputs_for_docs = []

        if 'majority_voting' in self.config.model_config.modules:
            # Try majority voting!

            generation_outputs_decoded = self.generator_tokenizer.batch_decode(generation_outputs, skip_special_tokens=True)

            generation_outputs = generation_outputs.reshape(batch_size, n_docs, -1)
            

            for b in range(batch_size):
                answer_proposals = generation_outputs_decoded[b*n_docs:(b+1)*n_docs]
                generation_outputs_for_docs.append(answer_proposals)
                counter = Counter()
                index_dict = {}
                for index, p in enumerate(answer_proposals):
                    index_dict.setdefault(p, index)
                counter = Counter(answer_proposals)
                top_proposals = [x[0] for x in counter.most_common(1)]
                top_cand_inds = [index_dict[p] for p in top_proposals]
                outputs.append(generation_outputs[b, top_cand_inds])

            outputs = torch.cat(outputs)
            doc_scores_log = -F.log_softmax(doc_scores, dim=-1)
            loss_with_doc_scores = doc_scores_log

        else:
            # Re-forward the generator, and use generation outputs as labels
            # obtain the loss of each (question, passage) pair

            # shift genereation results to left by one token
            # <bos> answer </s> --> answer </s> </s>(0)

            pad_token_id = self.generator.config.pad_token_id

            shifted_generation_outputs = torch.ones_like(generation_outputs) * pad_token_id
            shifted_generation_outputs[:, :-1] = generation_outputs[:, 1:]
            # print(self.generator_tokenizer.batch_decode(generation_outputs))
            # print('input:', generation_outputs)
            # print(self.generator_tokenizer.batch_decode(shifted_generation_outputs))
            # print('output:', shifted_generation_outputs)
            # input()
            forward_results = self.generator(
                                encoder_outputs=encoder_outputs, # use pre-computed encoder outputs
                                decoder_input_ids=generation_outputs,
                                return_dict=True)
            
            # Loss for each pair can be computed now
            logits = forward_results.logits

            # loss: batch_size x n_docs x seq_len
            loss_dict = self.get_loss(
                seq_logits=logits,
                doc_scores=doc_scores,
                target=shifted_generation_outputs, # use generation outputs as labels
                reduce_loss=False, # do not reduce loss
                exclude_bos_score=False,
                ignore_index=pad_token_id,
                n_docs=n_docs,
            )

            loss = loss_dict.nll_loss

            # decode the generation outputs
            generation_outputs_decoded = self.generator_tokenizer.batch_decode(generation_outputs, skip_special_tokens=True)

            # reshape generation_outputs
            generation_outputs = generation_outputs.reshape(batch_size, n_docs, -1)
            shifted_generation_outputs = shifted_generation_outputs.reshape(batch_size, n_docs, -1)
            

            if 'accumulated_loss_voting' in self.config.model_config.modules:
                # Loss: lower the better!
                doc_scores_log = -F.log_softmax(doc_scores, dim=-1)
                loss = doc_scores_log + (loss.sum(-1)) # batch_size x n_docs
                
                for b in range(batch_size):
                    loss_counter = defaultdict(int)
                    answer_proposals = generation_outputs_decoded[b*n_docs:(b+1)*n_docs]
                    index_dict = {}
                    for index, p in enumerate(answer_proposals):
                        index_dict.setdefault(p, index)
                    for proposal, proposal_loss in zip(answer_proposals, loss[b]):
                        loss_counter[proposal] = loss_counter[proposal] + proposal_loss
                    
                    sorted_counter = sorted(loss_counter.items(), key=lambda x: x[1])
                    # print(sorted_counter)
                    top_cand_inds = [index_dict[sorted_counter[0][0]]]

                    generation_outputs_for_docs.append(answer_proposals)
                    outputs.append(generation_outputs[b, top_cand_inds])

                outputs = torch.cat(outputs)

                loss_with_doc_scores = loss.sum(-1)

            elif 'loss_and_voting' in self.config.model_config.modules:

                n_cands = 5

                doc_scores_log = -F.log_softmax(doc_scores, dim=-1)
                loss_with_doc_scores = doc_scores_log + (loss.sum(-1))

                for b in range(batch_size):
                    # use topk to get indices of top candidates
                    top_cand_inds = (-loss_with_doc_scores[b]).topk(n_cands)[1]
                    answer_proposals = self.generator_tokenizer.batch_decode(generation_outputs[b, top_cand_inds], skip_special_tokens=True)

                    counter = Counter()
                    index_dict = {}
                    for index, p in enumerate(answer_proposals):
                        index_dict.setdefault(p, index)
                    counter = Counter(answer_proposals)
                    top_proposals = [x[0] for x in counter.most_common(1)]
                    top_cand_inds = [index_dict[p] for p in top_proposals]

                    outputs.append(generation_outputs[b, top_cand_inds])

                    generation_outputs_for_docs.append(answer_proposals)

                outputs = torch.cat(outputs)
                print(outputs)

            else:

                ################################
                # mean over tokens for each doc
                ################################
                # print('before g sum', loss)
                # print('before g sum', loss.shape)
                # mask = loss!=0
                # loss = (loss*mask).sum(dim=-1)/mask.sum(dim=-1)

                # print('after g sum', loss)
                # input()

                ################################
                # sum over tokens for each doc
                ################################
                # loss = loss.sum(-1)

                ################################
                # RAG thorough decoding sum over tokens for each doc
                # Currently having the best generalisation curve!
                ################################
                # doc_scores --> log_softmax --> log(g(z))
                # loss --> -log(p(y|x, z))
                # -log(g(z)p(y|x, z)) = -doc_scores + loss
                # batch_size x n_docs + batch_size x n_docs
                ################################


                doc_scores_log = -F.log_softmax(doc_scores, dim=-1)
                loss_with_doc_scores = doc_scores_log + (loss.sum(-1))
                # loss_with_doc_scores = loss.sum(-1)

                for b in range(batch_size):
                    # use topk to get indices of top candidates
                    top_cand_inds = (-loss_with_doc_scores[b]).topk(1)[1]

                    outputs.append(generation_outputs[b, top_cand_inds])
                    answer_proposals = generation_outputs_decoded[b*n_docs:(b+1)*n_docs]

                    generation_outputs_for_docs.append(answer_proposals)
                    # print(-loss[b])
                    # print(answer_proposals)

                outputs = torch.cat(outputs)



        return EasyDict(outputs=outputs, 
                        retrieved_docs=retrieved_docs, 
                        doc_scores=doc_scores.cpu().detach().numpy(),
                        loss_with_doc_scores=loss_with_doc_scores.cpu().detach().numpy(),
                        generation_outputs_for_docs=generation_outputs_for_docs)

    def get_loss(
        self, seq_logits, doc_scores, target, reduce_loss=True, epsilon=0.0, exclude_bos_score=False, ignore_index=-100, n_docs=None, retrieval_labels=None,
    ):
        """Compute loss

        Args:
            seq_logits (_type_): _description_
            doc_scores (_type_): _description_
            target (_type_): _description_
            reduce_loss (bool, optional): _description_. Defaults to True.
            epsilon (float, optional): _description_. Defaults to 0.0.
            exclude_bos_score (bool, optional): _description_. Defaults to False.
            ignore_index (int, optional): _description_. Defaults to -100.
            n_docs (_type_, optional): _description_. Defaults to None.
            retrieval_labels (_type_, optional): _description_. Defaults to None.

        Returns:
            EasyDict: every loss requested
        """

        if n_docs is None:
            n_docs = self.config.data_loader.additional.num_knowledge_passages
        
        loss_dict = EasyDict()
        
        # bos_token_id is None for T5
        bos_token_id = self.generator.config.bos_token_id
        use_bos = bos_token_id is not None and target[:, 0].eq(bos_token_id).all()

        
        batch_size = seq_logits.shape[0] // n_docs
        seq_len = seq_logits.shape[1]
        # seq_logits dim = (batch*n_docs, seq_len , #vocabs)
        seq_logprobs = nn.functional.log_softmax(seq_logits, dim=-1).view(
            batch_size, n_docs, -1, seq_logits.size(-1)
        )  # batch_size x n_docs x tgt_len x vocab_size
        doc_logprobs = nn.functional.log_softmax(doc_scores, dim=1).unsqueeze(-1).unsqueeze(-1)
        # print('doc_logprobs', doc_logprobs.shape)

        # RAG-sequence marginalization
        first_token_scores = seq_logprobs[:, :, :1, :]
        if use_bos:
            second_token_scores = seq_logprobs[:, :, 1:2, :]
            remainder = seq_logprobs[:, :, 2:, :]
            rag_logprobs = torch.cat([first_token_scores, second_token_scores + doc_logprobs, remainder], dim=2)
        else:
            # print('using T5 doc probs!')
            remainder = seq_logprobs[:, :, 1:, :]
            rag_logprobs = torch.cat([first_token_scores + doc_logprobs, remainder], dim=2)


        # Compute NLL Loss for seq_logprobs
        new_target = target.reshape(batch_size, n_docs, -1).unsqueeze(-1)
        assert new_target.dim() == seq_logprobs.dim()

        pad_mask = new_target.eq(ignore_index)

        if pad_mask.any() and ignore_index < 0:
            # fill -100 to be 0, avoid indexing error using gather
            new_target.masked_fill_(pad_mask, 0)

        ll = seq_logprobs.gather(dim=-1, index=new_target)
        if pad_mask.any():
            ll.masked_fill_(pad_mask, 0.0)
        
        ll = ll.squeeze(-1) # batch_size x n_docs x seq_len

        nll_loss = -ll
        loss_dict.nll_loss = nll_loss


        if self.config.model_config.loss_ratio.additional_loss != 0:
            if retrieval_labels is not None:
                first_token_scores = first_token_scores.detach()

                # batch_size x n_docs x voc_size
                first_token_scores = first_token_scores.squeeze(2)
                # batch_size x n_docs
                first_token_prediction = torch.argmax(first_token_scores, dim=-1)
                # print('first_token_prediction', first_token_prediction)

                # batch_size x n_docs
                first_token_target = target.reshape(batch_size, n_docs, -1)[:, :, 0]
                # print('first_token_target', first_token_target)

                # We found that matching the first token is a good approximation to prediction labels
                # So we use this to speed up
                prediction_labels = (first_token_prediction == first_token_target)
                # print(prediction_labels)
                retrieval_labels = retrieval_labels.to(seq_logits.device).float()
                # print(retrieval_labels)

                RAVQA_loss_type = self.config.model_config.RAVQA_loss_type
                if RAVQA_loss_type == 'Approach5':
                    ##############   approach 5:  ##################
                    # correct prediction + positive pseudo label = 1
                    # wrong prediction + positive pseudo label = -100
                    # correct prediction + negative pseudo label = -100
                    # wrong prediction + negative pseudo label = -100
                    merged_labels = torch.logical_and(prediction_labels, retrieval_labels).float()
                    ignore_mask = (merged_labels==0)
                
                elif RAVQA_loss_type == 'Approach6':
                    ##############   approach 6:  ##################
                    # correct prediction + positive pseudo label = 1
                    # wrong prediction + positive pseudo label = -100
                    # correct prediction + negative pseudo label = -100
                    # wrong prediction + negative pseudo label = 0

                    # prediction_res = prediction_labels[0][-1].unsqueeze(-1).unsqueeze(-1)

                    # good_labels =  torch.logical_and((prediction_labels[0][-1]==0),(prediction_res != prediction_labels))
                    # merged_labels =  torch.logical_and(good_labels,retrieval_labels).float()
                    # ignore_m = (prediction_res == prediction_labels)
                    # ignore_m = ignore_m | torch.logical_and((good_labels==0),(retrieval_labels==1))
                    # ignore_mask = ignore_m | torch.logical_and((good_labels==1),(retrieval_labels==0))

                    # merged_labels =  torch.logical_and((prediction_labels[0][-1]==0),(prediction_res != prediction_labels)).float()
                    # ignore_mask = (prediction_res == prediction_labels)


                    # input_loss = nll_loss.mean(-1)
                    # good_labels = torch.logical_and(((input_loss - input_loss[0][-1]) < 0),torch.logical_and(prediction_labels, retrieval_labels))
                    # bad_labels = torch.logical_and(((input_loss - input_loss[0][-1]) > 0),torch.logical_and((prediction_labels==0), (retrieval_labels==0)))
                    # ignore_mask = torch.logical_and(~good_labels, ~bad_labels)
                    # merged_labels = good_labels.float()

                    merged_labels = torch.logical_and(prediction_labels, retrieval_labels).float()
                    ignore_mask = torch.logical_or(
                        torch.logical_and((prediction_labels==0), (retrieval_labels==1)),
                        torch.logical_and((prediction_labels==1), (retrieval_labels==0)),
                        )
                elif RAVQA_loss_type == 'NoPR':
                    ##############   approach NoPR:  ##################
                    # correct prediction = 1
                    # wrong prediction = 0
                    merged_labels = prediction_labels.float()
                    ignore_mask = torch.zeros_like(merged_labels).bool().to(merged_labels.device)

                # doc_scores_softmaxed = F.softmax(doc_scores[:,:-1], dim=-1)
                # dist_loss = F.binary_cross_entropy(doc_scores_softmaxed, merged_labels[:,:-1], reduction='none')
                # dist_loss.masked_fill_(ignore_mask[:,:-1], 0.0)   

                doc_scores_softmaxed = F.softmax(doc_scores, dim=-1)
                dist_loss = F.binary_cross_entropy(doc_scores_softmaxed, merged_labels, reduction='none')
                dist_loss.masked_fill_(ignore_mask, 0.0)

                # remove_querys = 3
                # sort_res, _ = torch.sort(dist_loss,descending=True)#
                # res =dist_loss > sort_res[:,dist_loss.shape[1]-remove_querys]
                # # # res *= dist_loss < sort_res[:,2]
                # dist_loss *= res

                count_nonzero = torch.count_nonzero(dist_loss)
                if count_nonzero == 0:
                    dist_loss = 0
                else:
                    dist_loss = dist_loss.sum() / torch.count_nonzero(dist_loss)

                loss_dict.additional_loss = dist_loss
            else:
                loss_dict.additional_loss = 0


        if reduce_loss:

            mask = (pad_mask == 0)
            nll_loss = nll_loss.sum()
            nll_loss = nll_loss / torch.sum(mask)
            loss_dict.nll_loss = nll_loss



            # k_labels = merged_labels.clone()
            # # k_labels[:,0] = 1.0
            # if k_labels.sum() != 0:
            #     mask = (pad_mask == 0)
            #     nll_loss = (nll_loss*k_labels.unsqueeze(-1)).sum()
            #     nll_loss = nll_loss /  (mask.shape[-2] * k_labels.sum())
            #     loss_dict.nll_loss = nll_loss
            # else:
            #     loss_dict.nll_loss = (nll_loss*k_labels.unsqueeze(-1)).sum()


            # mask = (pad_mask == 0)
            # nll_loss = nll_loss[:,:-1,:].sum()
            # nll_loss = nll_loss / (torch.sum(mask) - mask.shape[-2] )
            # loss_dict.nll_loss = nll_loss
            

            # mask = (pad_mask == 0)
            # sort_res, _ = torch.sort(nll_loss.mean(-1) * (~res), descending=True)
            # res2 =nll_loss.mean(-1)  * (~res) > sort_res[:,nll_loss.shape[1]-remove_querys - res.sum()]
            # nll_loss *= (res2 | res).unsqueeze(-1)
            # nll_loss = nll_loss.sum()
            # nll_loss = nll_loss / (torch.sum(mask)-mask.shape[-2] * remove_querys)
            # loss_dict.nll_loss = nll_loss

            # nll_loss = nll_loss[:,:-1,:]
            # remove_querys = 1
            # mask = (pad_mask == 0)
            # sort_res, _ = torch.sort(nll_loss.mean(-1), descending=True)
            # res2 =nll_loss.mean(-1) > sort_res[:,nll_loss.shape[1]-remove_querys]
            # nll_loss *= res2.unsqueeze(-1)
            # nll_loss = nll_loss.sum()
            # nll_loss = nll_loss / (torch.sum(mask)-mask.shape[-2] * (remove_querys+1))
            # loss_dict.nll_loss = nll_loss


            # remove_querys = 2
            # mask = (pad_mask == 0)
            # sort_res, _ = torch.sort(nll_loss.mean(-1), descending=True)
            # res2 =nll_loss.mean(-1) > sort_res[:,nll_loss.shape[1]-remove_querys]
            # nll_loss *= res2.unsqueeze(-1)
            # nll_loss = nll_loss.sum()
            # nll_loss = nll_loss / (torch.sum(mask)-mask.shape[-2] * remove_querys)
            # loss_dict.nll_loss = nll_loss


        return loss_dict
        


    def get_retrieval_labels(self, 
                            batch_answers: List, 
                            batch_retrieved_docs: List):
        
        def most_frequent(List):
            return max(set(List), key = List.count)

        retrieved_docs = batch_retrieved_docs
        log_result = {
            'recall': [],
            'precision': [],
            'gold_precision': [],
            'gold_recall': [],
        }
        labels = []
        selected_answers = []
        for answer_list, docs in zip(batch_answers, retrieved_docs):
            
            filtered_answer_list = [ans for ans in answer_list if ans != '']
            gold_answer = most_frequent(filtered_answer_list)
            unique_answers = list(set(answer_list))
            counts = Counter(filtered_answer_list)
            answer_list_by_frequency = sorted(filtered_answer_list, key=lambda x: -counts[x])
            
            doc_texts = [doc['content'] for doc in docs]
            
            found_answers = []
            found_gold_answers = []

            
            if 'add_null_document' in self.config.model_config.modules:
                doc_texts = doc_texts[1:]

            this_batch_labels = [0] * len(doc_texts)
            K = len(doc_texts)
            
            for index, passage_data in enumerate(doc_texts):
                for answer in unique_answers:
                    if answer.lower() in passage_data.lower():
                        found_answers.append(answer)
                        this_batch_labels[index] = 1
                        break
                if gold_answer.lower() in passage_data.lower():
                    found_gold_answers.append(answer)
                    this_batch_labels[index] = 1

            for index, passage_data in enumerate(doc_texts):
                # by default the gold answer is selected, regardless the existence of answer
                selected_answer = gold_answer
                # select answer that appears in the document and with highest frequency
                if gold_answer.lower() in passage_data.lower():
                    pass # no change, by default the gold answer is selected
                else:
                    for answer in answer_list_by_frequency:
                        if answer == gold_answer:
                            continue # not consider gold answer
                        if answer.lower() in passage_data.lower():
                            selected_answer = answer
                            break
                selected_answers.append(selected_answer)

            labels.append(this_batch_labels)
                    
            if len(found_answers) > 0:
                # At least one answer is retireved
                log_result['recall'].append(1)
            else:
                log_result['recall'].append(0)
            # The proportion of retrieved knowledge has an answer
            log_result['precision'].append(len(found_answers) / K)

            if len(found_gold_answers) > 0:
                # if gold answer is found
                log_result['gold_recall'].append(1)
            else:
                log_result['gold_recall'].append(0)
            # The proportion of retrieved knowledge has the gold answer
            log_result['gold_precision'].append(len(found_gold_answers) / K)

        labels = torch.FloatTensor(labels)
        return EasyDict(
            retrieval_labels=labels,
            selected_answers=selected_answers,
        )