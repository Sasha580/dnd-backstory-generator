# DnD Character Backstory Generator

## Overview

This repository contains two related approaches to generating Dungeons & Dragons
character backstories from simple character descriptions (name, race, class):

1. A few-shot prompting pipeline using a Llama-style language model  
2. A GPT-2 fine-tuning experiment on the same DnD backstory dataset

Both approaches use public text data (character descriptions and backstories)
and focus on practical text generation for creative writing support.

---

## Project Structure

```text
.
├── Llama3_text_gen.ipynb        # Few-shot Llama-based backstory generation
├── GPT-2/
│   ├── custom_dataset.py        # PyTorch Dataset for GPT-2 fine-tuning
│   ├── training.py              # Training / inference script for GPT-2
│   └── epoch_2.pt              # Example fine-tuned GPT-2 checkpoint
├── requirements.txt
└── README.md
```


## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Sasha580/dnd-backstory-generator.git
   cd dnd-backstory-generator
   ```

2. Install necessary dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the project:

   - For the **few-shot Llama backstory generator**, start Jupyter and open the notebook:
     ```bash
     jupyter notebook
     ```
     Then open `Llama3_text_gen.ipynb` and run the cells.

   - For the **GPT-2 experiment**, run the training script:
     ```bash
     cd GPT-2
     python training.py
     ```

## Requirements

- Python 3.10+ (recommended)
- torch
- datasets
- transformers
- langchain
- jupyter
- llama3_package

## Approach

### Few-shot Llama backstory generator

The notebook `Llama3_text_gen.ipynb`:

- Loads a DnD backstory dataset using `datasets`.
- Cleans and normalizes the backstory text.
- Builds a set of few-shot examples (character description → backstory).
- Uses LangChain prompt templates to construct a single combined prompt.
- Calls a Llama-style language model to generate a new backstory for a user-defined character.

### GPT-2 fine-tuning experiment

The code in `GPT-2/`:

- Uses `datasets` to load the same DnD backstory data.
- Wraps the text in a custom `torch.utils.data.Dataset` (`custom_dataset.py`).
- Fine-tunes `GPT2LMHeadModel` from Hugging Face `transformers` using a simple training loop (`training.py`).
- Saves checkpoints (e.g. `epoch_2.pt`) and provides a helper to generate backstories
  from name, race and class with the fine-tuned model.



## Example (Llama backstory generation)

For example, for the character:

- Name: Kropus  
- Race: Tiefling  
- Class: Mage  

the Llama-based model generates the following backstory:

```text
Backstory: Kropus was born into a family of powerful sorcerers who had made a pact with 
a demon to increase their magical abilities. As a result, Kropus inherited some of this 
dark energy and became a skilled mage.

Growing up among his family's collection of ancient tomes and forbidden knowledge, Kropus 
became fascinated with the mysteries of the arcane arts. He spent countless hours studying 
and experimenting, mastering spells that would make even the most seasoned wizards jealous.
```

## Example (GPT-2 fine-tuned model)

For the same character:

- Name: Kropus  
- Race: Tiefling  
- Class: Mage  

the fine-tuned GPT-2 model often produces repetitive and partly incoherent text. For example:

```text
Backstory: Kropus was born to a Tiefling family in the countryside of the countryside. 
His father, a farmer, was a Tiefling, and his mother, a succubus, was a succubus. 
Kropus was born to a succubus and a Tiefling, respectively. His mother, a succubus, 
was a succubus who had been born there...
```

> This illustrates the limitations of the small GPT-2 baseline compared to the Llama few-shot approach, 
> which produces more coherent backstories.
    