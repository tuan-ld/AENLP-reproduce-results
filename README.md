# AENLP
This repository contains the implementation of NER inference demo from "Extraction of Comprehensive Drug Safety Information from Adverse Drug Event Narratives in Spontaneous Reporting System"

![figure1](https://user-images.githubusercontent.com/53844800/196883837-459ee966-e683-43f5-adc6-63d132999695.png)


# Implement inference_demo.py through streamlit
Step 1:
```
pip install -r requirements.txt
``` 
Step 2: 

download fine-tuned KAERS-BERT model configs [Google Drive link ](https://drive.google.com/drive/folders/1UioHU7Kg8fzSmzlOmvmt7kqtG6IhqygZ?usp=sharing)
then put it into the folder: ``` AENLP/model_configs/Link_for_downloading_model_configs.md ```

Step 3:
```
 streamlit run inference_demo.
```

# Example Usages
### Named entity recognition for extracting drug safety information
```python
from KAERSBERTforNER import inference_NER

ents = inference_NER('프라메딘 20mg 서방정을 2주간 복용한 환자에서 목 주변의 두드러기가 발생함.', 
                      print_result=True)
>>> 프라메딘/DrugCompound 20mg/DrugDose 서방정을/DrugRoAFormulation 2주간/DatePeriod 복용한 환자에서 
>>> 목/AdverseEvent 주변의/AdverseEvent 두드러기가/AdverseEvent 발생함.

print(ents)
>>> [{'text': '프라메딘', 'ent_type': 'DrugCompound', 'pos_start': 11, 'pos_end': 26}, 
>>> {'text': '20mg', 'ent_type': 'DrugDose', 'pos_start': 16, 'pos_end': 36},
>>> {'text': '서방정을', 'ent_type': 'DrugRoAFormulation', 'pos_start': 21, 'pos_end': 45},
>>> {'text': '2주간', 'ent_type': 'DatePeriod', 'pos_start': 25, 'pos_end': 53},
>>> {'text': '목 주변의 두드러기가', 'ent_type': 'AdverseEvent', 'pos_start': 46, 'pos_end': 96}]
```

# References
 - Kim, S., & Kang, T. (2022). Extraction of Comprehensive Drug Safety Information from Adverse Drug Event Narratives in Spontaneous Reporting System. (under submission)
