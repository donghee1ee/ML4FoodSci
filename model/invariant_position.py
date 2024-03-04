import torch

def compute_invariant_position_nontable(query_length, key_length, positions):
    """
    :param positions: the index of compound. 0 for prompt
    :return: attention_mask [batch_size, query_length, key_length]
    """ 

    context_position = torch.arange(query_length, dtype=torch.long)[:, None]
    memory_position = torch.arange(key_length, dtype=torch.long)[None, :]
    relative_position_template = memory_position - context_position

    relative_position_template = relative_position_template.unsqueeze(0).repeat(positions.shape[0], 1, 1).to(positions.device) ## (batch_size, 512, 512)

    # 1. prompt - prompt
    prompt_relative_position = relative_position_template.clone()
    prompt_mask = positions < 0.5
    prompt_mask = torch.bmm(torch.unsqueeze(prompt_mask.float(), 2), torch.unsqueeze(prompt_mask.float(), 1)) > 0.5
    prompt_relative_position = prompt_relative_position * prompt_mask

    # 2. chemical - chemical
    chem_relative_position = relative_position_template.clone()
    chem_mask = positions != 0
    chem_mask = torch.bmm(torch.unsqueeze(chem_mask.float(), 2), torch.unsqueeze(chem_mask.float(), 1)) > 0.5
    
    chem_diff = torch.abs(positions.unsqueeze(-1) - positions.unsqueeze(1))
    same_chem_mask = torch.logical_and(chem_diff < 0.5, chem_mask)
    diff_chem_mask = torch.logical_and(torch.logical_not(same_chem_mask), chem_mask)
    
    chem_relative_position = chem_relative_position * same_chem_mask + 512 * diff_chem_mask

    # 3. chemical - prompt / prompt - chemical
    bridge_mask = torch.logical_not(prompt_mask + chem_mask)
    bridge_relative_position = 512 * bridge_mask

    relative_position = prompt_relative_position + chem_relative_position + bridge_relative_position

    # TODO padding
    return relative_position






def compute_invariant_position_batch(batch_query_length, batch_key_length, batch_positions):
    batch_relative_position_matrices = []

    for positions in batch_positions:
        # Convert positions to a tensor for efficient processing
        positions_tensor = torch.tensor(positions)

        # Create a 2D grid of query and key indices
        query_indices = positions_tensor.unsqueeze(1).expand(-1, batch_key_length)
        key_indices = positions_tensor.unsqueeze(0).expand(batch_query_length, -1)

        # Compare the compound IDs of query and key indices
        same_compound_matrix = (query_indices == key_indices)

        # Handle padding: if either query or key is a padding token (-1), they are not in the same compound
        padding_mask = (query_indices == -1) | (key_indices == -1)
        same_compound_matrix[padding_mask] = False

        # Assign relative position based on whether they are in the same compound
        relative_position_matrix = torch.where(same_compound_matrix, torch.zeros_like(same_compound_matrix, dtype=torch.long), torch.ones_like(same_compound_matrix, dtype=torch.long))

        batch_relative_position_matrices.append(relative_position_matrix)

    return batch_relative_position_matrices


def get_compound_masks(length, positions):
    """
    Generate compound masks for a single example.

    Parameters:
    - length (int): The length of the sequence.
    - positions (list of tuples): Compound positions for a single example.

    Returns:
    - A dictionary containing compound masks for the example.
    """
    compound_masks = {}
    for compound_id, (start, end) in enumerate(positions):
        # Check for padding indicator and skip if found
        if start == -1 and end == -1:
            continue
        mask = [start <= idx <= end for idx in range(length)]
        compound_masks[compound_id] = mask
    return compound_masks

def compute_invariant_position(query_length, key_length, type_ids, row_ids, col_ids):
        """ Compute binned relative position bias for table"""
        # assume query_length == key_length

        context_position = torch.arange(query_length, dtype=torch.long)[:, None]
        memory_position = torch.arange(key_length, dtype=torch.long)[None, :]
        relative_position_template = memory_position - context_position  # shape (query_length, key_length)
        # shape of relative_position_template (batch_size, query_length, key_length)
        relative_position_template = relative_position_template.unsqueeze(0).repeat(type_ids.shape[0], 1, 1).to(
            type_ids.device)

        # relative position for meta data
        # others are set to 0
        meta_relative_position = relative_position_template.clone()
        meta_mask = torch.logical_and(type_ids < 2.5, type_ids > 0.5)  # shape (batch_size, query_length)
        # shape of meta_mask (batch_size, query_length, key_length)
        meta_mask = torch.bmm(torch.unsqueeze(meta_mask.float(), 2), torch.unsqueeze(meta_mask.float(), 1)) > 0.5
        meta_relative_position = meta_relative_position * meta_mask

        # relative position for cells
        # others are set to 0
        cell_relative_position = relative_position_template.clone()
        cell_mask = type_ids == 3  # shape (batch_size, query_length)
        # shape of cell_mask (batch_size, query_length, key_length)
        cell_mask = torch.bmm(torch.unsqueeze(cell_mask.float(), 2), torch.unsqueeze(cell_mask.float(), 1)) > 0.5

        row_diff = torch.abs(row_ids.unsqueeze(-1) - row_ids.unsqueeze(1))  # shape (batch_size, query_length, key_length)
        col_diff = torch.abs(col_ids.unsqueeze(-1) - col_ids.unsqueeze(1))  # shape (batch_size, query_length, key_length)

        same_cell_mask = torch.logical_and(row_diff + col_diff < 0.5, cell_mask)

        rc_cell_mask = torch.logical_and(torch.logical_not(same_cell_mask), cell_mask)

        cell_relative_position = cell_relative_position * same_cell_mask + 512 * rc_cell_mask

        # relative position between meta data and cell
        bridge_mask = torch.logical_not(meta_mask + cell_mask)  # 1 for attention between meta data and cell
        bridge_relative_position = 512 * bridge_mask

        # For a table:
        # A B
        # C D
        # where A for metadata, D for cells, B and C for attention between metadata and cells
        relative_position = meta_relative_position + cell_relative_position + bridge_relative_position

        return relative_position
