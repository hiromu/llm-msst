import collections
import csv
import enum
import json
import pathlib
import re

from langchain.llms import OpenAI, Replicate
from langchain.prompts import PromptTemplate
from langchain.schema import BaseOutputParser


class OutputParser(BaseOutputParser):
    def parse(self, text: str):
        match = re.search('([1-5])[.]?[):]', text)
        if match is None:
            print(f'Parse failed: {repr(text)}')
            return -1
        return match.group(1)


if __name__ == '__main__':
    scores = collections.defaultdict(lambda: collections.defaultdict(lambda: 0))

    template = PromptTemplate.from_template(
        '次の文を読んで、その答えとして一番よいと思うものを選んで下さい。\n' +
        '\n' +
        '{instruction_1}\n' +
        '\n' +
        '{instruction_2}\n' +
        '\n'.join([f'{index}) {{choice_{index}}}' for index in range(1, 6)])
    )
    parser = OutputParser()

    for model_name, model in [
        ('Llama 2 7B', Replicate(model='meta/llama-2-7b-chat:8e6975e5ed6174911a6ff3d60540dfd4844201974602551e10e9e87ab143d81e', model_kwargs={'temperature': 0.01, 'max_length': 500, 'top_p': 1})),
        ('Llama 2 13B', Replicate(model='meta/llama-2-13b-chat:f4e2de70d66816a838a89eeeb621910adffb0dd0baba3976c96980970978018d', model_kwargs={'temperature': 0.01, 'max_length': 500, 'top_p': 1})),
        ('Llama 2 70B', Replicate(model='meta/llama-2-70b-chat:02e509c789964a7ea8736978a43525956ef40397be9033abf9fd2badfe68c9e3', model_kwargs={'temperature': 0.01, 'max_length': 500, 'top_p': 1})),
        ('Dolly v2 12B', Replicate(model='replicate/dolly-v2-12b:ef0e1aefc61f8e096ebe4db6b2bacc297daf2ef6899f0f7e001ec445893500e5', model_kwargs={'temperature': 0.01, 'max_length': 500, 'top_p': 1})),
        ('GPT-3.5', OpenAI(model_name='gpt-3.5-turbo-0613', temperature=0)),
        ('GPT-4', OpenAI(model_name='gpt-4-0613', temperature=0)),
    ]:
        with open(pathlib.Path(__file__).parent / 'msst.csv') as input_csv:
            reader = csv.DictReader(input_csv)

            for index, question in enumerate(reader):
                category, answer = question.pop('category'), question.pop('answer')

                prompt = template.format(**question)
                response = model.predict(prompt)
                result = parser.parse(response)

                if result == answer:
                    scores[model_name][category] += 1
                print(f'- #{index + 1} ({category}): {result == answer}')

            print(f'{model_name}: metaphar = {scores[model_name]["metaphar"]}, sarcasm = {scores[model_name]["sarcasm"]}')

    with open(pathlib.Path(__file__).parent / 'results.json', 'w') as output_json:
        json.dump(scores, output_json)
