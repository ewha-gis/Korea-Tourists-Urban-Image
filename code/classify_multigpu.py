'''
MIT License

Copyright (c) 2021 GIS lab of EWHA.W.U

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

'''
import numpy as np
import datetime
import tensorflow as tf
import os
import csv
import multiprocessing
from multiprocessing import Array
import sys, getopt

imagePath = '/home/ubuntu/tmp2/2500test/test'                                      # 추론을 진행할 이미지 경로
modelFullPath = '/home/ubuntu/tmp2/output_graph.pb'                                      # 읽어들일 graph 파일 경로
labelsFullPath = '/home/ubuntu/tmp2/output_labels.txt'                              # 읽어들일 labels 파일 경로
outputFilePath = "/home/ubuntu/tmp2/result_temp"
total_gpu_count = 1
multi_pid = []
multi_count = []
multi_total = []

analyze_image_counts = 0

def main(argv): # 커맨드라인 매개변수 처리
    try:
        opts, etc_args = getopt.getopt(argv[1:], \
            "I:P:L:C:O:", ["image=", "pb=", "label=", "count=", "out="])
    except getopt.GetoptError:
        print('-I <이미지 경로> -P <그래프 경로> -L <라벨 경로> -C <전체 GPU 코어 개수> -O <산출물 경로>')
        sys.exit(2)

    for opt, arg in opts:
        if opt in ("-I", "--image"): # image path
            global imagePath 
            imagePath = arg
        elif opt in ("-P", "--pb"): # pb path
            global modelFullPath 
            modelFullPath = arg
        elif opt in ("-L", "--label"): # label path
            global labelsFullPath 
            labelsFullPath = arg
        elif opt in ("-C", "--count"): # total gpu count
            global total_gpu_count 
            total_gpu_count = int(arg)
            global multi_pid
            global multi_count
            global multi_total
            multi_pid = Array('i', range(0, total_gpu_count, 1))
            multi_count = Array('i', range(0, total_gpu_count, 1))
            multi_total = Array('i', range(0, total_gpu_count, 1))

        elif opt in ("-O", "--out"): # output path
            global outputFilePath 
            outputFilePath = arg
    
def create_graph(): # 저장된 pb파일에서 그래프 생성
    """저장된(saved) GraphDef 파일로부터 graph를 생성하고 saver를 반환한다."""
    # 저장된(saved) graph_def.pb로부터 graph를 생성한다.
    with tf.gfile.FastGFile(modelFullPath, 'rb') as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')

def load_images(target_gpu): # 이미지 불러오기
    imagenames = []
    for filepath in tf.io.gfile.glob(os.path.join(imagePath, '*.jpg'))[:40000]: #주어진 패턴과 일치하는 파일 목록을 변환        
        imagenames.append(filepath)

    print("전체 이미지 개수: ", len(imagenames))
    global analyze_image_counts
    analyze_image_counts = (len(imagenames)//total_gpu_count + ((len(imagenames)%total_gpu_count > target_gpu) and 0 or -1))
    print("처리 이미지 할당량: ", analyze_image_counts)
    multi_total[target_gpu] = analyze_image_counts
    return imagenames

def run_inference_on_image(imagename, device): # 이미지 분류하기
    answer = None
    dic = {}
    i=1
    
    if not tf.io.gfile.exists(imagename):
        tf.logging.fatal('File does not exist %s', imagename)
        return answer
        
    image_data = tf.gfile.FastGFile(imagename, 'rb').read()
    gpu_options = tf.compat.v1.GPUOptions(visible_device_list=device)
    
    with tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options)) as sess:
        #print('Image: %s, Process: #%s' % (imagename, device))
        softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
        predictions = sess.run(softmax_tensor, {'DecodeJpeg/contents:0': image_data})
        predictions = np.squeeze(predictions)
        top_k = predictions.argsort()[-5:][::-1]  # 가장 높은 확률을 가진 5개(top 5)의 예측값(predictions)을 얻는다.
        f = open(labelsFullPath, 'rb')
        lines = f.readlines()
        labels = [str(w).replace("\n", "") for w in lines]
        
        for node_id in top_k:            
            human_string = labels[node_id]
            score = predictions[node_id]
            #print('%s (score = %.5f)' % (human_string, score))
            
            key = str(i)
            dic['top'+key]= human_string #.딕셔너리 생성(top accuracy를 기록하기 위함)
            dic['score'+key]= score
            i = i+1
            
        answer = labels[top_k[0]]
        return dic 

def saveDataToCSV(results) : # csv에 분류 결과 저장하기
    with open(outputFilePath, "a+", newline='') as file: # csv 생성 경로 지정
        f = csv.writer(file)
        f.writerow(["ID","name", "top1", "score1","top2","score2","top3","score3","top4","score4","top5","score5" ])
        for imagenames, data in results:
            index = 0
            for photo in data :
                f.writerow([index, imagenames[index], photo['top1'], photo['score1'], photo['top2'], photo['score2'], photo['top3'], photo['score3'], photo['top4'], photo['score4'], photo['top5'], photo['score5']])
                index = index+1

def logging():
    length = 33
    print('=' * length, flush= True)
    print('|{0: ^9}|{1: ^5}|{2: ^15}|'.format("PID", "GPU", "CURRENT_USAGE"))
    print('-' * length)
    for idx in range(0, total_gpu_count):
        print('|{0: ^9}|{1: ^5}|{2: >7}/{3: >7}|'.format(multi_pid[idx], idx, multi_count[idx], multi_total[idx]))
    print('-' * length)

def multiProc(target_gpu): # 멀티 프로세스
    data = []
    names = []
    idx = 0
    anal_count = 0
    multi_pid[target_gpu] = os.getpid()
    # 저장된(saved) GraphDef 파일로부터 graph를 생성한다.
    create_graph()
    for imagename in load_images(target_gpu): # 폴더 내에 있는 이미지 경로 불러오기
        idx += 1
        # print(idx % total_gpu_count)
        if idx % total_gpu_count == int(target_gpu):
            data.append(run_inference_on_image(imagename, str(target_gpu))) # 해당하는 이미지를 분류한 결과를 data 셋에 저장
            names.append(imagename)
            multi_count[target_gpu] = anal_count
            anal_count += 1
            logging()
    return names, data

if __name__ == '__main__': #메인
    t1 = datetime.datetime.now()
    print("START TIME: {0}".format(t1))
    procs = []
    main(sys.argv)
    pool = multiprocessing.Pool(processes=total_gpu_count)
    results = pool.map(multiProc, range(0, total_gpu_count)) #데이터 csv로 저장
    saveDataToCSV(results)
    t2 = datetime.datetime.now()
    print("START  TIME: {0}".format(t1))
    print("END    TIME: {0}".format(t2))
    print("ELAPSE TIME: {0} sec".format((t2 - t1).total_seconds()))
