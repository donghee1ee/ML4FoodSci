from typing import Callable, List, Optional, Tuple, Union
import torch

from transformers import T5ForConditionalGeneration, RagTokenForGeneration, RagSequenceForGeneration, PretrainedConfig, PreTrainedModel, RagRetriever

class ConstrainedT5(T5ForConditionalGeneration):
    def __init__(self, config, constraint_ids):
        super().__init__(config)
        # This should be a list of token ids that you want to allow.
        self.constraint_ids = constraint_ids

    # def prepare_inputs_for_generation(self, input_ids, **kwargs):
    #     return {'input_ids': input_ids, 'attention_mask': kwargs.get('attention_mask')}

    def enforce_constraints(self, next_token_logits):
        constraint_indices_tensor = torch.tensor(self.constraint_ids, device=next_token_logits.device, dtype=torch.long)

        # Create a full tensor of -inf values with the same size as next_token_logits
        constraints = torch.full_like(next_token_logits, float('-inf'))

        # Now scatter the values from next_token_logits into constraints tensor
        # For each position in the batch and sequence length, take the logits for the allowed indices
        # This requires creating an index tensor that has the same size along the batch and sequence dimensions
        batch_size, seq_length, _ = next_token_logits.size()
        expanded_indices = constraint_indices_tensor.unsqueeze(0).unsqueeze(0).expand(batch_size, seq_length, -1)

        # Use the expanded index tensor to scatter the logits
        constraints.scatter_(dim=2, index=expanded_indices, src=next_token_logits.gather(dim=2, index=expanded_indices))

        return constraints

    def __call__(self, *args, **kwargs):
        # Run the normal forward pass
        output = super().__call__(*args, **kwargs)

        # Enforce constraints on the logits
        output['logits'] = self.enforce_constraints(output.logits)

        return output

class CustomRagTokenForGeneration(RagTokenForGeneration):
    def __init__(
        self,
        config: Optional[PretrainedConfig] = None,
        question_encoder: Optional[PreTrainedModel] = None,
        generator: Optional[PreTrainedModel] = None,
        retriever: Optional[RagRetriever] = None,
        constraint_ids: Optional[List[int]] = None,
        n_docs = None,
        **kwargs,
    ):
        if n_docs is not None:
            config.n_docs = n_docs # default 5
        super().__init__(config, question_encoder=question_encoder, generator=generator, retriever=retriever, **kwargs)

        self.constraint_ids = constraint_ids

    def enforce_constraints(self, next_token_logits):
        constraint_indices_tensor = torch.tensor(self.constraint_ids, device=next_token_logits.device, dtype=torch.long)

        # Create a full tensor of -inf values with the same size as next_token_logits
        constraints = torch.full_like(next_token_logits, float('-inf'))

        # Now scatter the values from next_token_logits into constraints tensor
        # For each position in the batch and sequence length, take the logits for the allowed indices
        # This requires creating an index tensor that has the same size along the batch and sequence dimensions
        batch_size, seq_length, _ = next_token_logits.size()
        expanded_indices = constraint_indices_tensor.unsqueeze(0).unsqueeze(0).expand(batch_size, seq_length, -1)

        # Use the expanded index tensor to scatter the logits
        constraints.scatter_(dim=2, index=expanded_indices, src=next_token_logits.gather(dim=2, index=expanded_indices))

        return constraints
    
    def __call__(self, *args, **kwargs):
        # Run the normal forward pass
        output = super().__call__(*args, **kwargs)

        # Enforce constraints on the logits
        if self.constraint_ids is not None:
            output['logits'] = self.enforce_constraints(output.logits)

        return output

class CustomRagSequenceForGeneration(RagSequenceForGeneration):
    def __init__(
        self,
        config: Optional[PretrainedConfig] = None,
        question_encoder: Optional[PreTrainedModel] = None,
        generator: Optional[PreTrainedModel] = None,
        retriever: Optional[RagRetriever] = None,
        constraint_ids: Optional[List[int]] = None,
        n_docs = None,
        **kwargs,
    ):
        if n_docs is not None:
            config.n_docs = n_docs # default 5
        super().__init__(config, question_encoder=question_encoder, generator=generator, retriever=retriever, **kwargs)

        self.constraint_ids = constraint_ids

    def enforce_constraints(self, next_token_logits):
        constraint_indices_tensor = torch.tensor(self.constraint_ids, device=next_token_logits.device, dtype=torch.long)

        # Create a full tensor of -inf values with the same size as next_token_logits
        constraints = torch.full_like(next_token_logits, float('-inf'))

        # Now scatter the values from next_token_logits into constraints tensor
        # For each position in the batch and sequence length, take the logits for the allowed indices
        # This requires creating an index tensor that has the same size along the batch and sequence dimensions
        batch_size, seq_length, _ = next_token_logits.size()
        expanded_indices = constraint_indices_tensor.unsqueeze(0).unsqueeze(0).expand(batch_size, seq_length, -1)

        # Use the expanded index tensor to scatter the logits
        constraints.scatter_(dim=2, index=expanded_indices, src=next_token_logits.gather(dim=2, index=expanded_indices))

        return constraints
    
    def __call__(self, *args, **kwargs):
        # Run the normal forward pass
        output = super().__call__(*args, **kwargs)

        # Enforce constraints on the logits
        if self.constraint_ids is not None:
            output['logits'] = self.enforce_constraints(output.logits)

        return output