import os
from PIL import Image
import numpy as np
from sklearn.decomposition import PCA
from sklearn import svm as svmlib


def dataload(path):
    all_images = []
    all_labels = []
    i = 0
    for train_test_path_name in os.listdir(path):
        label = 0
        if train_test_path_name == '.DS_Store':
            continue
        train_test_path = os.path.join(path, train_test_path_name)
        #//print(train_test_path) # './class4-facedata/test' './class4-facedata/train'
        for student_path_name in os.listdir(train_test_path):
            if student_path_name == '.DS_Store':
                continue            
            student_path = os.path.join(train_test_path, student_path_name)
            for image_path_name in os.listdir(student_path):
                if image_path_name == '.DS_Store':
                    continue
                all_labels.append(label)

                image_path = os.path.join(student_path, image_path_name)
                #"./class4-facedata/test/guosiqi/guosiqi.jpg-15.jpg"
                img_gray = Image.open(image_path).convert('L')
                img_np = np.array(img_gray)
                img_np = img_np.reshape((1, -1))
                #//print(img_np.shape) # (1, 62500)
                if i==0:
                    all_images = img_np
                else:
                    all_images = np.concatenate((all_images,img_np),axis=0)
                i+=1
            label += 1 # label correspond to one person
    return all_images, all_labels


def pca_function(all_images, components_value):
    # 实现降维功能，返回得到的降维数组
    model = PCA(n_components=components_value)
    components = model.fit_transform(all_images)
    
    return components   


def svm_poly(X_train, X_test, Y_train, Y_test, cc):
    # 实现svm分类功能，得到分类结果
    clf = svmlib.SVC(C=cc, kernel='poly',decision_function_shape='ovr')
    clf.fit(X_train, Y_train)
    
    trainAcc = clf.score(X_train, Y_train)
    testAcc = clf.score(X_test, Y_test)
    print('training prediction:%.3f' % trainAcc)
    print('test data prediction:%.3f' % testAcc) 

def svm_linear(X_train, X_test, Y_train, Y_test, cc):
    clf = svmlib.SVC(C=cc, kernel='linear',decision_function_shape='ovr')
    clf.fit(X_train, Y_train)    
    trainAcc = clf.score(X_train, Y_train)
    testAcc = clf.score(X_test, Y_test)
    print('training prediction:%.3f' % trainAcc)
    print('test data prediction:%.3f' % testAcc) 

def svm_rbf(X_train, X_test, Y_train, Y_test, cc):
    clf = svmlib.SVC(C=cc, kernel='rbf',decision_function_shape='ovr')
    clf.fit(X_train, Y_train)    
    trainAcc = clf.score(X_train, Y_train)
    testAcc = clf.score(X_test, Y_test)
    print('training prediction:%.3f' % trainAcc)
    print('test data prediction:%.3f' % testAcc) 

def svm_sig(X_train, X_test, Y_train, Y_test, cc):
    clf = svmlib.SVC(C=cc, kernel='sigmoid',decision_function_shape='ovr')
    clf.fit(X_train, Y_train)    
    trainAcc = clf.score(X_train, Y_train)
    testAcc = clf.score(X_test, Y_test)
    print('training prediction:%.3f' % trainAcc)
    print('test data prediction:%.3f' % testAcc) 



if __name__ == "__main__":
    data_path = './class4-facedata' # 文件路径
    all_images, all_labels = dataload(data_path) #@ all_images: (1020, 250*250), all_labels: 1020_[51 people]
    
    print("########## Case 1 : Components Int ###########")
    for eachComponent in [1,5,10,20,30,40,50]:
        # 降维
        pca_images = pca_function(all_images, eachComponent)

        # 划分训练集和测试集
        train_num = 714  # 训练集样本数
        X_train = pca_images[:714, :]
        X_test = pca_images[714:, :]
        Y_train = all_labels[:714]
        Y_test = all_labels[714:]
        
        # 训练SVM分类
        print("############### n_components: ", eachComponent)
        print("############### SVM poly: ###############")
        for cc in [0.0001, 0.001, 0.01, 0.1, 1, 10, 100]:
            print("C: ", cc)
            svm_poly(X_train, X_test, Y_train, Y_test, cc)
        
        if eachComponent > 10: 
            print("############### SVM linear: ###############")
            for cc in [0.0001, 0.001, 0.01, 0.1, 1, 10, 100]:
                print("C: ", cc)
                svm_linear(X_train, X_test, Y_train, Y_test, cc)   
                
        print("############### SVM rbf: ###############")
        for cc in [0.0001, 0.001, 0.01, 0.1, 1, 10, 100]:
            print("C: ", cc)
            svm_rbf(X_train, X_test, Y_train, Y_test, cc)     
        
        print("############### SVM sigmoid: ###############")
        for cc in [0.0001, 0.001, 0.01, 0.1, 1, 10, 100]:
            print("C: ", cc)
            svm_sig(X_train, X_test, Y_train, Y_test, cc)   
            
            
    print("########## Case 2 : Components Float ###########")
    for eachComponent in [0.2, 0.4, 0.6, 0.8, 0.95]:
        # 降维
        pca_images = pca_function(all_images, eachComponent)

        # 划分训练集和测试集
        train_num = 714  # 训练集样本数
        X_train = pca_images[:714, :]
        X_test = pca_images[714:, :]
        Y_train = all_labels[:714]
        Y_test = all_labels[714:]
        
        # 训练SVM分类
        print("############### n_components: ", eachComponent)
        print("############### SVM poly: ###############")
        for cc in [0.0001, 0.001, 0.01, 0.1, 1, 10, 100]:
            print("C: ", cc)
            svm_poly(X_train, X_test, Y_train, Y_test, cc)
        
        if eachComponent > 10: 
            print("############### SVM linear: ###############")
            for cc in [0.0001, 0.001, 0.01, 0.1, 1, 10, 100]:
                print("C: ", cc)
                svm_linear(X_train, X_test, Y_train, Y_test, cc)   
                
        print("############## SVM rbf: ###############")
        for cc in [0.0001, 0.001, 0.01, 0.1, 1, 10, 100]:
            print("C: ", cc)
            svm_rbf(X_train, X_test, Y_train, Y_test, cc)     
        
        print("############### SVM sigmoid: ###############")
        for cc in [0.0001, 0.001, 0.01, 0.1, 1, 10, 100]:
            print("C: ", cc)
            svm_sig(X_train, X_test, Y_train, Y_test, cc)   