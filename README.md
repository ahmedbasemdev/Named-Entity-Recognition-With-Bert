# Named-Entity-Recognition-With-Bert

Named-entity recognition (NER) is a subtask of information extraction that seeks to locate and classify named entity mentioned
in unstructured text into pre-defined categories such as person names, organizations, locations,
medical codes, time expressions, quantities, monetary values, percentages, etc.

# Tools
- Python 
- PyTorch
- Bert Transformer pre-trained

# Training    
1. Modules    
`pip install transformers`    
- Download Dataset    
  `!gdown 1MqhuH7pGn9tcIHSkHwLZomAIVh4aqO-7`    
- move dataset into data folder   
`!mkdir data`        
`!mv ner_dataset.csv data/ner_dataset.csv`    
- traning model     
`!python train.py`      

# Testing 
`!python predict.py "this is egypt"`





