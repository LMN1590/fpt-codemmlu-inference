from typing import Dict
def parse_LLM_output_to_valid_JSON(output:str)->str:
    return output.replace("\n"," ").replace("\t"," ")

def check_key(dict_item:Dict,key:str):
    return (dict_item is not None and key in dict_item and dict_item[key] is not None)