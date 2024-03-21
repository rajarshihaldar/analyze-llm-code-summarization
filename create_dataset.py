from llm_datasets.codexglue import CodeXGlue

def get_codexglue(remove_func=False, only_func=False, remove_comments=True, adversarial_func=False, remove_keyword=False, debug=False):
    root = './'
    data_split = 'test'
    use_code_tokens = True
    if remove_keyword:
        use_code_tokens = False

    dataset = CodeXGlue(root, dataset=data_split, remove_func=remove_func,
        only_func=only_func, remove_comments=remove_comments, adversarial_func=adversarial_func,
        remove_keyword=remove_keyword, use_code_tokens=use_code_tokens)

    if debug:
        code, desc = dataset[10]
        print(code)
        if remove_keyword:
            new_code = code.split()
            new_code.insert(1, '""" <insert> """')
            new_code = ' '.join(new_code)
        else:
            new_code = code.replace(desc, ' <insert> ')
            new_code = new_code.replace('<insert> \n\t', '<insert> ')
            new_code = new_code.replace('<insert> \n    ', '<insert> ')
        print(new_code)
    return dataset

if __name__=="__main__":
    get_codexglue(debug=True)