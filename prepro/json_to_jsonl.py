import json


def convert_jsonl(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as json_file:
        data = json.load(json_file)

    with open(output_file, 'w', encoding='utf-8') as jsonl_file:
        for item in data['documents']:
            jsonl_file.write(json.dumps(item, ensure_ascii=False) + '\n')

if __name__ == '__main__':
    input_file = '/storage/hjchoi/Document_Summary_text/Validation/law_valid_original/valid_original.json'
    output_file = '/storage/hjchoi/Document_Summary_text/Validation/law.jsonl'
    convert_jsonl(input_file, output_file)