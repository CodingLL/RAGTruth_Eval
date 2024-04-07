def hallucination_check_prompt(task_type, source_info, response):
    if task_type == 'Summary':
        article = source_info
        prompt = 'Below is the original news:\n'
        prompt += f'{article}'+'\n'
        prompt += 'Below is a summary of the news:\n'
        prompt += f'{response}'+'\n'
        prompt += 'Your task is to identify and label any hallucinated statements in the summary that are unsupported or contradicted by the original news. '
    elif task_type == 'Data2txt':
        business_info = source_info
        prompt = 'Below is a structured data in the JSON format:\n'
        prompt += f'{business_info}'+'\n'
        prompt += 'Below is an overview article written in accordance with the structured data:\n'
        prompt += f'{response}'+'\n'
        prompt += 'Your task is to identify and label any hallucinated statements in the overview that are unsupported or contradicted by the structured data. '
    elif task_type == 'QA':
        question, passages = source_info['question'], source_info['passages']
        prompt = 'Below is a question:\n'
        prompt += f'{question}'+'\n\n'
        prompt += 'Below are related passages:\n'
        prompt += f'{passages}'+'\n'
        prompt += 'Below is an answer:\n'
        prompt += f'{response}'+'\n\n'
        prompt += 'Your task is to identify and label any hallucinated statements in the answer that are unsupported or contradicted by the passages. '
    prompt += 'Then, compile the labeled hallucinated spans into a JSON list, with each list item representing a separate hallucinated span.\n'
    prompt += 'Output:'
    return prompt
