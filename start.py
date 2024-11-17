
# from train_sketch import Coach_sketch
# from train_bayer import  Coach_sketch
# from train_rgbformer import Coach_sketch
from train_sid import Coach_sketch

if __name__=='__main__':
    num = 1
    # coach = Coach_sketch(train_num=0,device=0,mode='train')
    coach = Coach_sketch(num=num,device=num,mode='train')
    # coach = Coach_sketch(train_num=0, device=0, mode='test')###test只能cuda0
    coach.train()
    # coach.test_image_generation()

