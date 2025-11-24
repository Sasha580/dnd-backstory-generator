import re
from torch.utils.data import Dataset


MAX_LENGTH = 200


class CustomDataset(Dataset):
    """
    A PyTorch Dataset Subclass to preprocess and tokenize text data for language modeling tasks.

    Attributes:
        tokenizer: A tokenizer object for text tokenization.
        tokenized_data: A list of tokenized chunks prepared for model input.

    Methods:
        tokenize_and_chunk(data):
            Tokenizes input data by splitting it into chunks of sentences 
            and processing them with the tokenizer to create input_ids and attention_masks.

        __len__():
            Returns the number of samples in the dataset.

        __getitem__(idx):
            Returns tokenized data (input_ids and attention_mask) for the specified index.
    """

    def __init__(self, data, tokenizer):
        """
        Initializes the CustomDataset object.

        Args:
            data (list): A list of text strings to be tokenized.
            tokenizer (object): A tokenizer object used to process the text data.
        """

        self.tokenizer = tokenizer
        self.tokenized_data = self.tokenize_and_chunk(data)

    def tokenize_and_chunk(self, data):
        tokenized_chunks = []
        attention_masks = []
        sentences_step = 2  # Number of sentences per chunk
        for text in data:
            sentences = re.split(r'(?<=[.!?]) +', text)  # Split text into sentences
            for i in range(0, len(sentences), sentences_step):
                chuck = ''.join(sentences[i:i + sentences_step])  # Combine sentences for the chunk
                tokenized_chunk = self.tokenizer(
                    chuck, 
                    return_tensors='pt', 
                    padding="max_length", 
                    truncation=True, 
                    max_length=MAX_LENGTH)
                tokenized_chunks.append(tokenized_chunk)
        return tokenized_chunks

    def __len__(self):
        """
        Returns the number of tokenized chunks in the dataset.

        Returns:
            int: The number of samples in the dataset.
        """

        return len(self.tokenized_data)

    def __getitem__(self, idx):
        """
        Retrieves the tokenized data for a specific index.

        Args:
            idx (int): The index of the desired data item.

        Returns:
            tuple: A tuple containing input_ids and attention_mask for the specified index.
        """

        return self.tokenized_data[idx]['input_ids'], self.tokenized_data[idx]['attention_mask']