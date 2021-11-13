"""
@Fire
https://github.com/fire717
"""
import os
import random
import pandas as pd   
import torch

from lib import init, Data, MoveNet, Task

from config import cfg





def main(cfg):

    init(cfg)


    model = MoveNet(num_classes=cfg["num_classes"],
                    width_mult=cfg["width_mult"],
                    mode='train')
    


    run_task = Task(cfg, model)
    run_task.modelLoad('output/test/e100_valacc0.98349.pth')


    run_task.model.eval()
    run_task.model.to("cuda")

    #data type nchw
    dummy_input1 = torch.randn(1, 3, 192, 192).to("cuda")
    input_names = [ "input1"] #自己命名
    output_names = [ "output1","output2","output3","output4" ]
    # torch.onnx.export(model, (dummy_input1, dummy_input2, dummy_input3), "C3AE.onnx", verbose=True, input_names=input_names, output_names=output_names)
    torch.onnx.export(run_task.model, dummy_input1, "output/pose.onnx", 
        verbose=True, input_names=input_names, output_names=output_names,
        do_constant_folding=True,opset_version=11)


    # model = MoveNet(num_classes=cfg["num_classes"],
    #                 width_mult=cfg["width_mult"],
    #                 mode='test')
    
    

    # run_task = Task(cfg, model)
    # run_task.modelLoad('output/test/e104_valacc0.95586.pth')


    # run_task.model.eval()
    # run_task.model.to("cuda")

    # #data type nchw
    # dummy_input1 = torch.randn(1, 3, 192, 192).to("cuda")
    # input_names = [ "input1"] #自己命名
    # output_names = [ "output1" ]
    # # torch.onnx.export(model, (dummy_input1, dummy_input2, dummy_input3), "C3AE.onnx", verbose=True, input_names=input_names, output_names=output_names)
    # torch.onnx.export(run_task.model, dummy_input1, "output/pose.onnx", 
    #     verbose=True, input_names=input_names, output_names=output_names,
    #     do_constant_folding=True,opset_version=11)



if __name__ == '__main__':
    main(cfg)









