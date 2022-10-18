from argparse import ArgumentParser

def get_args():
    parser =ArgumentParser()

    parser.add_argument("-epochs", type=int, default=50, dest = "epochs", help="number of epochs")
    parser.add_argument("-batch_size", type=int, default=64,dest="batch_size")
    parser.add_argument("-lr",type=float,default=0.001,help="learning rate")

    parser.add_argument("-input_dim",type=int,default=768)
    parser.add_argument("-hidden_dim",type=int,default=128)
    parser.add_argument("-output_dim",type=int,default=2)

    args = parser.parse_args()

    return args

