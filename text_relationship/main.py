# -*- coding:utf-8 -*-

import os
import numpy as np
import torch
import argparse
import math

# Attribute name
ATTRIBUTE_WORD = [
    "2D Boxy","3D Boxy","Round","Vert Cyl","Horiz Cyl",
    "Occluded","Tail","Beak","Head","Ear",
    "Snout","Nose","Mouth","Hair","Face",
    "Eye","Torso","Hand","Arm","Leg",
    "Foot/Shoe","Wing","Propeller","Jet engine","Window",
    "Row Wind","Wheel","Door","Headlight","Taillight",
    "Side mirror","Exhaust","Pedal","Handlebars","Engine",
    "Sail","Mast","Text","Label","Furniture Leg",
    "Furniture Back","Furniture Seat","Furniture Arm","Horn","Rein",
    "Saddle","Leaf","Flower","Stem/Trunk","Pot",
    "Screen","Skin","Metal","Plastic","Wood",
    "Cloth","Furry","Glass","Feather","Wool",
    "Clear","Shiny","Vegetation","Leather"
]

# Split
CLASS_NAME_SPLIT1 = np.array(["aeroplane","bicycle","boat","bottle","car",
                              "cat","chair","dining table","dog","horse",
                              "person","potted plant","sheep","train","tv monitor",
                              "bird","bus","cow","motorbike","sofa"])

CLASS_NAME_SPLIT2 = np.array(["bicycle","bird","boat","bus","car",
                              "cat","chair","dining table","dog","motorbike","person","potted plant","sheep","train","tv monitor",
                              "aeroplane","bottle","cow","horse","sofa"])

CLASS_NAME_SPLIT3 = np.array(["aeroplane","bicycle","bird","bottle","bus",
                              "car","chair","cow","dining table","dog",
                              "horse","person","potted plant","train","tv monitor",
                              "boat","cat","motorbike","sheep","sofa"])

# class name
# CLASS_NAME = [
#     "aeroplane",
#     "bicycle",
#     "bird",
#     "boat",
#     "bottle",
#     "bus",
#     "car",
#     "cat",
#     "chair",
#     "cow",
#     "dining table", # Original is diningtable
#     "dog",
#     "horse",
#     "motorbike",
#     "person",
#     "potted plant", # Original is pottedplant
#     "sheep",
#     "sofa",
#     "train",
#     "tv monitor"    # Original is tvmonitor
# ]

# The class of each attribute.
CLASS_VEC = {
    "aeroplane":[51,79,17,6,130,70,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,165,57,83,98,70,103,60,6,8,0,26,0,0,0,0,0,88,0,0,0,0,0,0,0,0,0,0,0,0,0,0,178,44,2,0,0,87,0,0,0,105,0,0],
    "bicycle":[0,0,0,0,0,105,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,149,0,6,4,1,0,108,117,0,0,0,45,0,0,0,0,0,0,0,0,0,0,0,0,0,0,154,70,0,0,0,0,0,0,0,77,0,0],
    "bird":[0,0,0,0,0,104,168,211,233,0,0,0,0,0,0,184,223,0,0,158,135,156,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,247,0,0,0,0,0],
    "boat":[80,92,9,26,66,94,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,64,51,1,18,0,0,1,2,0,0,10,56,70,42,0,0,0,0,0,0,0,0,0,0,0,0,0,0,118,73,109,0,0,0,0,0,0,42,0,0],
    "bottle":[0,0,0,257,29,103,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,194,0,0,0,0,0,0,0,0,0,0,0,0,0,7,87,0,0,0,187,0,0,83,112,0,0],
    "bus":[0,61,0,0,0,38,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,65,65,51,34,40,13,36,3,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,73,0,0,0,0,65,0,0,0,45,0,0],
    "car":[0,358,0,0,0,364,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,434,0,343,297,222,182,247,51,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,499,0,0,0,0,0,0,0,0,346,0,0],
    "cat":[0,0,0,0,0,87,85,0,184,179,168,0,0,0,0,153,152,0,0,135,119,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,181,0,0,0,0,0,0,0],
    "chair":[156,214,0,0,0,312,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,230,389,255,143,0,0,0,0,0,0,0,0,0,94,101,242,176,0,0,0,0,0,68,0,0],
    "cow":[0,0,0,0,0,69,32,0,67,59,60,0,0,0,0,43,75,0,0,68,39,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,19,13,0,0,0,0,0,0,0,0,0,0,0,96,0,0,0,0,0,0,0],
    "dining table":[54,0,34,0,0,84,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,35,0,0,0,0,0,0,0,0,0,0,0,0,16,19,71,30,0,3,0,0,5,22,0,0],
    "dog":[0,0,0,0,0,130,110,0,233,225,213,0,0,0,0,196,209,0,0,195,159,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,231,0,0,0,0,0,0,0],
    "horse":[0,0,0,0,0,80,69,0,141,126,135,0,0,0,0,120,129,0,0,131,110,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,78,59,0,0,0,0,0,0,0,0,0,0,143,0,0,0,0,0,0,0],
    "motorbike":[0,0,0,0,0,82,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,132,0,72,38,68,54,0,125,70,0,0,56,0,0,0,0,0,0,0,0,0,0,0,0,0,0,145,94,0,0,0,0,0,0,0,100,0,0],
    "person":[0,0,0,0,0,1702,0,0,2163,1305,0,1521,1421,1665,1590,1378,1872,1333,1734,1232,700,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2155,0,0,0,2272,0,0,0,0,0,0,0,0],
    "potted plant":[0,0,0,179,13,72,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,200,66,102,194,0,0,0,0,0,0,0,0,0,0,0,0,215,0],
    "sheep":[0,0,0,0,0,63,40,0,91,75,70,0,0,0,0,57,100,0,0,78,52,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,117,0,0,0,0],
    "sofa":[35,76,10,0,0,93,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,31,102,102,90,0,0,0,0,0,0,0,0,0,0,6,32,106,0,0,0,0,0,0,0,14],
    "train":[0,72,9,6,44,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,66,45,45,44,39,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,89,0,0,0,0,0,0,0,0,51,0,0],
    "tv monitor":[87,56,0,0,0,56,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,125,0,41,122,0,0,0,128,0,0,0,60,0,0],
}

def mkdir(name):
    '''
    Create folder
    '''
    isExists=os.path.exists(name)
    if not isExists:
        os.makedirs(name)
    return None

def is_number(s):
    """
    Judge if the string belong to number
    """
    try:
        float(s)
        return True
    except ValueError:
        pass
 
    return False

def read_glove_840B_300d():
    """
    Get the word embedding from txt
    """
    embeddings_dict = {}

    with open("./pre-trained/glove.840B.300d.txt", 'r', encoding="utf-8") as f:
        for line in f:
            values = line.split()

            for i in range(10):
                word = values[i]
                if is_number(word):
                    key_loc = i
                    break
            
            word_list = values[:key_loc]
            word = " ".join(i for i in word_list)

            vector = np.asarray(values[key_loc:], "float32")
            embeddings_dict[word] = vector
    print("Successfully loading the pre-trained text: glove.840B.300d.txt")
    return embeddings_dict

def get_attribute_vector(embeddings_dict, method = "average"):
    """
    Embedding the Word Vector
      embeddings_dict: word2vec, using GloVe embedding
    """
    attribute_word_embedding = {}

    for attribute_word_ in ATTRIBUTE_WORD:
        # Judge if special charactor in String
        if " " in attribute_word_:
            attribute_word_embedding[attribute_word_] = (embeddings_dict[attribute_word_.split(" ")[0]] + embeddings_dict[attribute_word_.split(" ")[1]])/2
        elif "/" in attribute_word_:
            attribute_word_embedding[attribute_word_] = (embeddings_dict[attribute_word_.split("/")[0]] + embeddings_dict[attribute_word_.split("/")[1]])/2
        else:
            attribute_word_embedding[attribute_word_] = embeddings_dict[attribute_word_]
    print("Successfully embedding the attribute word.")
    return attribute_word_embedding

def main(args):
    """
    Main function, compute the relationship between the object using text.
    """
    # which split
    if args.split == "split1":
        CLASS_NAME = CLASS_NAME_SPLIT1
    elif args.split == "split2":
        CLASS_NAME = CLASS_NAME_SPLIT2
    elif args.split == "split3":
        CLASS_NAME = CLASS_NAME_SPLIT3

    embeddings_dict = read_glove_840B_300d()

    if args.method == "class_word":
        word_vec = []

        # Memorize the feature vector
        for class_name in CLASS_NAME:
            if " " not in class_name:
                word_vec.append(embeddings_dict[class_name])
            else:
                word_vec.append((embeddings_dict[class_name.split(" ")[0]] + embeddings_dict[class_name.split(" ")[1]])/2)

        word_vec = np.array(word_vec)

        assert word_vec.shape[0]==20

        # word_vec_norm = torch.nn.functional.normalize(torch.Tensor(word_vec), p=2, dim=1)

        # similarity = torch.mm(word_vec_norm, word_vec_norm.T)
        
        # # Ensure the dimension is true.
        # assert similarity.shape == torch.Size([len(CLASS_NAME),len(CLASS_NAME)])

        # similarity = torch.clamp(similarity,-1,1)   # prevent nan
        # similarity = 1 - torch.arccos(similarity) /  math.pi

        # # Ensure the dimension is true.
        # assert similarity.shape == torch.Size([len(CLASS_NAME),len(CLASS_NAME)])

        # # Save
        # mkdir(os.path.join("./text_relationship_txt/", args.split))
        # np.savetxt(os.path.join("./text_relationship_txt/", args.split) + "/" + args.method + "-" + args.split+".txt", similarity)
        # mkdir(os.path.join("./text_relationship_matrix", args.split))
        # np.save(os.path.join("./text_relationship_matrix", args.split) + "/"+args.method+"-"+args.split+".npy", similarity)

    elif args.method == "attribute_word":
        # Attribute word embedding
        attribute_word_embedding = get_attribute_vector(embeddings_dict)

        word_vec = []

        for class_name in CLASS_NAME:
            # attribute vector embedding
            attribute_vec = []
            for i in range(len(ATTRIBUTE_WORD)):
                if CLASS_VEC[class_name][i] != 0:
                    attribute_vec.append(attribute_word_embedding[ATTRIBUTE_WORD[i]])
                else:
                    attribute_vec.append(embeddings_dict["null"])
            # reshape to one hot    
            attribute_vec = np.array(attribute_vec).reshape([-1])
            
            # assert the dimention
            assert len(attribute_vec.shape) == 1

            word_vec.append(attribute_vec)
        
        word_vec = np.array(word_vec)
        assert word_vec.shape[0]==20

    ## This is the public code for all methods
    word_vec_norm = torch.nn.functional.normalize(torch.Tensor(word_vec), p=2, dim=1)

    similarity = torch.mm(word_vec_norm, word_vec_norm.T)
    
    # Ensure the dimension is true.
    assert similarity.shape == torch.Size([len(CLASS_NAME),len(CLASS_NAME)])

    similarity = torch.clamp(similarity,-1,1)   # prevent nan
    similarity = 1 - torch.arccos(similarity) /  math.pi

    # Ensure the dimension is true.
    assert similarity.shape == torch.Size([len(CLASS_NAME),len(CLASS_NAME)])

    # Save
    mkdir(os.path.join("./text_relationship_txt/", args.split))
    np.savetxt(os.path.join("./text_relationship_txt/", args.split) + "/" + args.method + "-" + args.split+".txt", similarity)
    mkdir(os.path.join("./text_relationship_matrix", args.split))
    np.save(os.path.join("./text_relationship_matrix", args.split) + "/"+args.method+"-"+args.split+".npy", similarity)
    
def parse_args():
    parser = argparse.ArgumentParser(description='VOC 2008 datasets, text relationship.')
    parser.add_argument('--split', type=str,
        default="split1",
        choices=["split1","split2","split3"],
        help='which set.')
    parser.add_argument('--method', type=str,
        default="class_word",
        choices=["attribute_word","class_word"],
        help='which method to get relationship.')
    
    args = parser.parse_args()

    return args

if __name__ == "__main__":
    args = parse_args()
    main(args)

