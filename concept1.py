import numpy as np
import pandas as pd
import os


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LinearRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


f_path = './[배포용]+MINIST+Data/'
files = [os.path.join(f_path, f) for f in os.listdir(f_path)]


data_set_y = []
data_set_x = []


for f in files:
    y = int(f.split('/')[-1].split('_')[0])
    x = pd.read_csv(f, header=None).values
    data_set_x.append(x)
    data_set_y.append(y)

df_y = pd.DataFrame(data_set_y, columns=['label'])
data_set_x = np.array(data_set_x)


def polynomial_basis(x, degree):
    """
    x: 입력 데이터
    degree: 다항식 차수
    """
    x = np.array(x)

    # 다항식 차수에 따라 다항식 생성
    polynomial = np.ones((x.shape[0], 1))
    for i in range(1, degree+1):
        polynomial = np.concatenate([polynomial, x**i], axis=1)

    return polynomial



def feature_and_basis(temp_image):

    ## num basis ## 
    num_basis = 5

    ###################### 특징 1, 2 생성 ######################
    ## 가로축 projection 후 기대 값, 분산 계산 ##
    projection_v = np.sum(temp_image, axis=1)
    pdf_v = projection_v / np.sum(projection_v, keepdims=True)
    expected_v = np.sum(projection_v * pdf_v, keepdims=True)
    variance_v = np.sum((projection_v - expected_v)**2 * pdf_v)
    expected_v = expected_v.reshape(-1)

    ###################### 특징 3, 4 생성 ######################
    ## 세로축 projection 후 기대 값, 분산 계산 ##

    projection_h = np.sum(temp_image, axis=0)
    pdf_h = projection_h / np.sum(projection_h, keepdims=True)
    expected_h = np.sum(projection_h * pdf_h,  keepdims=True)
    variance_h = np.sum((projection_h - expected_h)**2 * pdf_h)
    expected_h = expected_h.reshape(-1)


    features = np.vstack([expected_v, variance_v, expected_h, variance_h])


    #### polynomial basis function ####
    feature_out = polynomial_basis(x=features, degree=num_basis)
    
    basis_fea_set = np.resize(feature_out,(1, 4*num_basis+1))
 
    return basis_fea_set



all_data_x = np.array([feature_and_basis(data).reshape(-1) for data in data_set_x])

all_data_x.shape


# data_split : 전체 data 분할 및 label 설정하는 함수
def data_split(input_x, input_y):
    """ 
    """

    train_test_dict = dict()
    data_x = input_x
    y = input_y.values if isinstance(input_y, pd.DataFrame) else input_y

    X_train, X_test, y_train, y_test = train_test_split(data_x, y, test_size=0.2, random_state=2020,shuffle=True)

    # scaler = StandardScaler()
    # scaler = scaler.fit(X_train)

    # X_train = scaler.transform(X_train)
    # X_test = scaler.transform(X_test)

    train_test_dict['test_x'] = X_test
    train_test_dict['test_y'] = y_test
    train_test_dict['train_x'] = X_train
    train_test_dict['train_y'] = y_train

    return train_test_dict

train_test_dict = data_split(input_x=all_data_x, input_y=df_y)


model = LinearRegression()
model.fit(train_test_dict['train_x'], train_test_dict['train_y'])



y_pred = model.predict(train_test_dict['test_x'])


y_pred[y_pred < 0.5] = 0
y_pred[(0.5<=y_pred) & (y_pred < 1.5)] = 1
y_pred[1.5<=y_pred] = 2



print('acc',accuracy_score(y_true=train_test_dict['test_y'], y_pred=y_pred))

print(classification_report(y_true=train_test_dict['test_y'], y_pred=y_pred))
confusion_matrix(y_true=train_test_dict['test_y'], y_pred=y_pred)


weight = pd.DataFrame(model.coef_.T,columns=['weight'])
weight


weight.to_csv('학번_weight.csv',index=False)


## 검증 코드 검증##
x_test = np.array([], dtype='float32')
x_test = np.resize(x_test, (0, 21))
print(x_test.shape)
for d in data_set_x:
    temp=feature_and_basis(d)
    x_test = np.concatenate((x_test, temp), axis=0)


x_test.shape


data_x = feature_and_basis(temp_image=data_set_x[0])