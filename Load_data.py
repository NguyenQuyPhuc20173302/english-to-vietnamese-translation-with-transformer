def load_data(file_name):
    list_sentence = []
    with open(file_name) as f:
        for line in f.readlines():
            list_sentence.append(line)

    return list_sentence


def text_pairs(en, vn):
    text_pair = []
    for i in range(len(vn)):
        text_pair.append((en[i].split('\n')[0], "[start] " + vn[i].split('\n')[0] + " [end]"))

    return text_pair


def process_data():
    vn_train = load_data('./data/train/train.vi')
    en_train = load_data('./data/train/train.en')

    vn_test = load_data('./data/test/test.tgt')
    en_test = load_data('./data/test/test.src')

    vn_val = load_data('./data/val/valid.tgt')
    en_val = load_data('./data/val/valid.src')

    train_pairs = text_pairs(en_train, vn_train)
    test_pairs = text_pairs(en_test, vn_test)
    val_pairs = text_pairs(en_val, vn_val)

    print(f"{len(train_pairs)} training pairs")
    print(f"{len(val_pairs)} validation pairs")
    print(f'{len(test_pairs)} test pairs')


process_data()