# AENLP
This repository contains the implementation of NER inference demo from "Extraction of Comprehensive Drug Safety Information from Adverse Drug Event Narratives in Spontaneous Reporting System"

![figure1](https://user-images.githubusercontent.com/53844800/196883837-459ee966-e683-43f5-adc6-63d132999695.png)


# Implement inference_demo.py through streamlit
``` 
pip install -r requirements.txt
streamlit run inference_demo.py
```

#### Example Usage
```python
import spacy

from scispacy.abbreviation import AbbreviationDetector

nlp = spacy.load("en_core_sci_sm")

# Add the abbreviation pipe to the spacy pipeline.
nlp.add_pipe("abbreviation_detector")

doc = nlp("Spinal and bulbar muscular atrophy (SBMA) is an \
           inherited motor neuron disease caused by the expansion \
           of a polyglutamine tract within the androgen receptor (AR). \
           SBMA can be caused by this easily.")

print("Abbreviation", "\t", "Definition")
for abrv in doc._.abbreviations:
	print(f"{abrv} \t ({abrv.start}, {abrv.end}) {abrv._.long_form}")

>>> Abbreviation	 Span	    Definition
>>> SBMA 		 (33, 34)   Spinal and bulbar muscular atrophy
>>> SBMA 	   	 (6, 7)     Spinal and bulbar muscular atrophy
>>> AR   		 (29, 30)   androgen receptor

# References
 - Kim, S., & Kang, T. (2022). Extraction of Comprehensive Drug Safety Information from Adverse Drug Event Narratives in Spontaneous Reporting System. (under submission)
