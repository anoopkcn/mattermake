import torch
from typing import Dict, List, Optional, Callable
from torch.utils.data import Dataset
from mattermake.utils.pylogger import get_pylogger


class CrystalSequenceDataset(Dataset):
    """Dataset for tokenized crystal structures with composition-first approach"""

    def __init__(
        self,
        data: List[Dict],
        transform: Optional[Callable] = None,
        filter_fn: Optional[Callable] = None,
        max_sequence_length: Optional[int] = None,
        cache_tokenized: bool = True,
    ):
        """
        Initialize the dataset with preprocessed crystal structure data

        Args:
            data: List of dictionaries containing tokenized structure data
            transform: Optional transform to apply to the data
            filter_fn: Optional function to filter data items
            max_sequence_length: Override max sequence length from tokenizer
            cache_tokenized: Whether to cache tokenized sequences in memory
        """
        self.logger = get_pylogger(__name__)
        self.transform = transform
        self.cache_tokenized = cache_tokenized
        self.max_sequence_length = max_sequence_length

        if filter_fn is not None:
            self.data = [item for item in data if filter_fn(item)]
        else:
            self.data = data

        self.material_ids = [item["material_id"] for item in self.data]
        self.formulas = [item["formula"] for item in self.data]
        self.space_groups = [item.get("space_group") for item in self.data]

        if cache_tokenized:
            self.logger.info(
                f"Caching tokenized sequences for {len(self.data)} structures..."
            )
            self.cached_items = []
            for i, item in enumerate(self.data):
                tokens, segment_ids, masks, targets = self._prepare_sequence(item)
                self.cached_items.append(
                    {
                        "input_ids": tokens,
                        "segment_ids": segment_ids,
                        "masks": masks,
                        "target_ids": targets,
                        "material_id": item["material_id"],
                        "composition": item["token_data"].composition,
                    }
                )
            self.logger.info("Done caching sequences")

    def _prepare_sequence(self, item):
        """Prepare token sequences (without padding to max length)"""
        token_data = item["token_data"]

        max_length = self.max_sequence_length

        if max_length and len(token_data.sequence) > max_length:
            tokens = token_data.sequence[:max_length]
            segment_ids = token_data.segment_ids[:max_length]
            masks = {k: v[:max_length] for k, v in token_data.masks.items()}
        else:
            tokens = token_data.sequence
            segment_ids = token_data.segment_ids
            masks = token_data.masks

        tokens = torch.tensor(tokens, dtype=torch.long)
        segment_ids = torch.tensor(segment_ids, dtype=torch.long)
        masks = {k: torch.tensor(v, dtype=torch.bool) for k, v in masks.items()}

        targets = torch.zeros_like(tokens)
        targets[:-1] = tokens[1:].clone()
        pad_token = 2  # Assuming pad token is 2
        targets[-1] = pad_token

        return tokens, segment_ids, masks, targets

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """Get a data item with tokenized sequences"""
        if self.cache_tokenized:
            item = self.cached_items[idx]

            if self.transform is not None:
                item = self.transform(item)

            return item

        data_item = self.data[idx]
        tokens, segment_ids, masks, targets = self._prepare_sequence(data_item)

        item = {
            "input_ids": tokens,
            "segment_ids": segment_ids,
            "masks": masks,
            "target_ids": targets,
            "material_id": data_item["material_id"],
            "composition": data_item["token_data"].composition,
        }

        if self.transform is not None:
            item = self.transform(item)

        return item

    def get_structure(self, idx):
        """Get the original structure for a given index"""
        return self.data[idx]["structure"]

    def get_statistics(self):
        """Get dataset statistics"""
        space_groups = [sg for sg in self.space_groups if sg is not None]
        return {
            "num_structures": len(self.data),
            "unique_formulas": len(set(self.formulas)),
            "unique_space_groups": len(set(space_groups)),
            "space_group_distribution": {
                sg: space_groups.count(sg) for sg in set(space_groups)
            },
            "formula_distribution": {
                f: self.formulas.count(f) for f in set(self.formulas)
            },
        }

    def filter_by_space_group(self, space_group):
        """Create a new dataset filtered by space group"""

        def filter_fn(item):
            return item.get("space_group") == space_group

        return CrystalSequenceDataset(
            self.data,
            transform=self.transform,
            filter_fn=filter_fn,
            max_sequence_length=self.max_sequence_length,
            cache_tokenized=self.cache_tokenized,
        )


def collate_crystal_sequences(batch):
    """Custom collation function for handling variable-length crystal sequences"""
    # Find max length in this batch
    max_length = max(item["input_ids"].size(0) for item in batch)

    # Initialize batch dictionaries
    batched = {
        "input_ids": [],
        "segment_ids": [],
        "target_ids": [],
        "material_id": [],
        "composition": [],
    }

    # Handle masks if present
    if "masks" in batch[0] and batch[0]["masks"]:
        batched["masks"] = {k: [] for k in batch[0]["masks"].keys()}

    # Process each item in the batch
    for item in batch:
        # Get current sequence length
        cur_len = item["input_ids"].size(0)
        padding_len = max_length - cur_len

        # Pad sequences if needed
        if padding_len > 0:
            # Pad input_ids with PAD_TOKEN (2)
            input_ids = torch.cat(
                [
                    item["input_ids"],
                    torch.full(
                        (padding_len,),
                        2,
                        dtype=torch.long,
                        device=item["input_ids"].device,
                    ),
                ]
            )

            # Pad segment_ids with SPECIAL (0)
            segment_ids = torch.cat(
                [
                    item["segment_ids"],
                    torch.zeros(
                        padding_len, dtype=torch.long, device=item["segment_ids"].device
                    ),
                ]
            )

            # Pad target_ids with PAD_TOKEN (2)
            target_ids = torch.cat(
                [
                    item["target_ids"],
                    torch.full(
                        (padding_len,),
                        2,
                        dtype=torch.long,
                        device=item["target_ids"].device,
                    ),
                ]
            )
        else:
            input_ids = item["input_ids"]
            segment_ids = item["segment_ids"]
            target_ids = item["target_ids"]

        # Add to batch
        batched["input_ids"].append(input_ids)
        batched["segment_ids"].append(segment_ids)
        batched["target_ids"].append(target_ids)
        batched["material_id"].append(item["material_id"])
        batched["composition"].append(item["composition"])

        # Handle masks if present
        if "masks" in item and item["masks"]:
            for k, v in item["masks"].items():
                if padding_len > 0:
                    padded_mask = torch.cat(
                        [v, torch.zeros(padding_len, dtype=torch.bool, device=v.device)]
                    )
                else:
                    padded_mask = v
                batched["masks"][k].append(padded_mask)

    # Stack tensors
    batched["input_ids"] = torch.stack(batched["input_ids"])
    batched["segment_ids"] = torch.stack(batched["segment_ids"])
    batched["target_ids"] = torch.stack(batched["target_ids"])

    # Stack masks if present
    if "masks" in batched and batched["masks"]:
        for k in batched["masks"].keys():
            batched["masks"][k] = torch.stack(batched["masks"][k])

    return batched
