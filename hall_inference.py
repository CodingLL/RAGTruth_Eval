from huggingface_hub import InferenceClient, AsyncInferenceClient
from argparse import ArgumentParser
from tqdm import tqdm
import os, json, asyncio
from hall_prompt import hallucination_check_prompt

def load_data(prefix: str):
    source_id2info = {}
    source_id2task = {}
    filepath = os.path.join(prefix, 'source_info.jsonl')
    with open(filepath) as f:
        for line in f:
            dic = json.loads(line)
            source_id2info[dic['source_id']] = dic['source_info']
            source_id2task[dic['source_id']] = dic['task_type']
        
    resp_data =[]
    filepath = os.path.join(prefix, 'response.jsonl')
    with open(filepath) as f:
        for line in f:
            dic = json.loads(line)
            if dic['split'] == 'train':
                continue
            dic['source_info'] = source_id2info[dic['source_id']]
            dic['task_type'] = source_id2task[dic['source_id']]
            dic['input_str'] = hallucination_check_prompt(dic['task_type'], 
                                                          source_info=dic['source_info'], 
                                                          response=dic['response'])
            resp_data.append(dic)
    return resp_data
    

async def generate_response(args, sample, cnt, client):
    output = await client.text_generation(sample['input_str'], 
                                          max_new_tokens=512, 
                                          stream=False,
                                          do_sample=True if args.temp != None else False,
                                          temperature=args.temp,
                                          )
    output = output.strip()
    if cnt % args.print_freq == 0:
        print('\ncnt: {}, id: {}, source_id: {}, task_type: {}, model: {}'\
            .format(cnt, sample['id'], sample['source_id'], sample['task_type'], sample['model']))
        print('input: \n', sample['input_str'])
        print('output:\t', output)
        print('label: \t', sample['labels'])
        
    sample['output'] = output
    return sample

async def main(args):
    # loading data
    print('test_file: {}'.format(args.test_filepath))
    data_list = load_data(args.test_filepath)
    # sort the data by prompt length to inference faster
    data_list = sorted(data_list, key=lambda x: len(x['input_str'])) 
    print('size of data: {}'.format(len(data_list)))
    
    final_data = []
    tasks = []
    client = AsyncInferenceClient(model=args.client, timeout=100)
    
    # inference asynchronously
    for cnt in tqdm(range(len(data_list))):
        tasks.append(asyncio.create_task(generate_response(args, data_list[cnt], cnt, client)))
        if len(tasks)==args.parallel:
            print('start running count {}'.format(cnt))
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for i in range(len(results)):
                ret = results[i]
                final_data.append(ret)
            tasks = []

    results = await asyncio.gather(*tasks, return_exceptions=True)
    for i in range(len(results)):
        ret = results[i]
        final_data.append(ret)
        
    # save output
    print('results saved in : {}'.format(args.output_file))
    with open(args.output_file, 'w') as f:
        for d in final_data:
            f.write(json.dumps(d)+"\n")
            

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--test_filepath', type=str, default='./data/')
    parser.add_argument('--output_file', default=f'./infer_res/llama2_13b_lora_0218.jsonl')
    parser.add_argument('--parallel', type=int, default=20)
    parser.add_argument('--client', type=str, default="http://localhost:8321")
    parser.add_argument('--print_freq', type=int, default=20)
    parser.add_argument('--temp', type=float, default=None)
    
    args = parser.parse_args()
    asyncio.run(main(args))