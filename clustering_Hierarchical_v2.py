# coding: utf-8
import argparse
import os
import pandas as pd
import seaborn as sns
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.neighbors import DistanceMetric
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
from scipy.cluster import hierarchy
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import fcluster
from results import FitResult, PredictResult
import cvthtml
from model_param import ModelParam

from preprocess import Preprocess

class Hierachical:
    _IMAGE_PATH = './{0}'
    _IMPORTANCE_IMG_PATH = _IMAGE_PATH.format('feature_importances.png')
    _CONF_MATRIX_FIT_IMG_PATH = _IMAGE_PATH.format('confusion_matrix_fit.png')
    _CONF_MATRIX_PRED_IMG_PATH = _IMAGE_PATH.format('confusion_matrix_pred.png')
    _ROC_IMG_PATH = _IMAGE_PATH.format('ROC_Curvers.png')

    # 폰트 설정
    plt.rcParams['font.family'] = 'NanumGothic'
    # matplotlib.rcParams['font.family'] = 'AppleGothic'
    plt.rcParams['font.size'] = 10 # 글자 크기
    plt.rcParams['axes.unicode_minus'] = False # 한글 폰트 사용 시, 마이너스 글자가 깨지는 현상을 해결
    def __init__(self, param):
        self.param = param
        self.param: ModelParam
        self.data_path = self.param.data_path
        self._target_data_path = None
        self.org_data_set = None
        self.x_train = None
        self.y_train = None
        self.x_val = None
        self.y_val = None
        self.x = None
        self.y = None
        self._fit_result = None
        self._fit_result: FitResult
        self._pred_result = None
        self._pred_result: PredictResult


    def preprocess(self):
        """ 2단계: 데이터 전처리 후 모델 학습
            _preprocess: 데이터 전처리
            _fit: 모델 학습

        Returns:
            FitResult(): 모델학습 결과
        """
        # 데이터 불러오기
        dataset = pd.read_csv(self.param.data_path)
        # 데이터 전처리
        self._preprocess(dataset)
        # 모델 학습
        result_fit = self.fit()

        return result_fit

    def _preprocess(self, dataset: pd.DataFrame(), is_predict=False):
        """입력된 데이터의 전처리 프로세스
            2단계, 4단계에서 모두 실행
            result는 X, y 분리된 데이터셋 (4개 or 2개)

        Args:
            dataset (pd.DataFrame): 사용자가 업로드한 데이터 파일. X, y 값을 가짐
            is_fit (bool, optional): 모델 fit 여부를 결정. Defaults to True.
        """
        prep = Preprocess(self.param, dataset, is_predict)
        if not is_predict:
            self._fit_result = prep.process()
        else:
            self._pred_result = prep.pred_process()


    def fit(self):

        """계층적 클러스터링 시작
        Args:
            input_path : 학습에 사용할 입력 파일명
            x_name_list : 학습에 사용할 컬럼 목록
            target_name : 군집 레이블로 사용할 컬럼
            k : 군집의 갯수
            rotate : 차추 출력 방향
        """

        self.x_train, self.y_train = self._fit_result.x_train, self._fit_result.y_train
        self.x_val, self.y_val = self._fit_result.x_val, self._fit_result.y_val

        x_labels = self.param.independent_values
        new_idx_list_train = self.y_train
        features_vector_train = self.x_train

        """
        ***군집에 사용할 컬럼를 기반으로한 데이터를 기반한 제곱 유클리디안 거리를 이용하여 인접 메트릭스를 생성한다.
        """
        pd.options.display.float_format = '{:,.3f}'.format

        print('<h3>1. 제조데이터 계층적 군집분석</h3>')
        print('<h5>&nbsp;&nbsp;1) 유클리디안 거리 계산 결과</h3>')
        print('<h5>&nbsp;&nbsp;■ 근접행렬</h5>')
        print(
            "<style> .dataframe {width:unset;min-width:700px} \n.dataframe th {padding:3px} \n .dataframe td {padding:3px;text-align:right}</style>")
        adjacency_matrix_tr = euclidean_distances(features_vector_train)
        adjacentmatrix_df_tr = pd.DataFrame(adjacency_matrix_tr)
        adjacentmatrix_df_val = pd.DataFrame(self.y_val)

        rename_dict = {}
        for idx, column_name in enumerate(new_idx_list_train):
            rename_dict[idx] = column_name
        adjacentmatrix_df_tr.rename(rename_dict, axis='index', inplace=True)
        adjacentmatrix_df_tr.rename(rename_dict, axis='columns', inplace=True)
        # index_list = adjacentmatrix_df.index.tolist()
        # adjacentmatrix_df['구분'] = index_list
        # adjacentmatrix_df.set_index('구분', inplace=True)
        print(adjacentmatrix_df_tr.to_html(justify='center').replace('<th></th>', '<th>구분</th>'))
        # self.render_mpl_table(adjacentmatrix_df, header_columns=0, col_width=2.0)
        # imgstr = cvthtml.plt2html()
        # plt.cla()
        # print(imgstr)
        print("&nbsp;&nbsp;※ 거리 값이 작을수록 가까운(같은) 군집일 가능성이 크다는 의미이며, 거리가 클수록 먼(다른) 군집일 가능성이 크다는 것을 의미합니다.<br>")
        """
        ***인접 메트리스를 이용하여 덴드로그램을 생성/화면에 출력한다.
        """
        # metric 수정(임시)
        print('<br><h5>&nbsp;&nbsp;■ 완전 연결법(최장거리)을 사용한 덴드로그램</h5>')
        # 20201203 요청사항 수정
        # 20201204 요청사항 수정(x축 레이블 상단 배치), 거리 스케일 1에서 25로 변경.
        self.model = hierarchy.linkage(adjacentmatrix_df_tr, metric='euclidean', method='complete')
        self.model[:, 2] = np.sqrt(self.model[:, 2])
        self.model[:, 2] = np.interp(self.model[:, 2], (self.model[:, 2].min(), self.model[:, 2].max()), (1, 25))

        def get_mx_lbl_length(labelList):
            mx_length = 0
            for lbl in labelList:
                if len(str(lbl)) > mx_length:
                    mx_length = len(str(lbl))
            return mx_length

        labelList = adjacentmatrix_df_tr.columns.to_list()
        height = new_idx_list_train.shape[0]
        if height > 1000:
            height = 300
        fig, ax = plt.subplots(figsize=(25, height), dpi=50, constrained_layout=True)
        mx_lbl_length = 0.04 + (get_mx_lbl_length(labelList) / 500)
        # plt.subplots_adjust(left=mx_lbl_length,bottom=0.002,right=0.9, top=0.9)
        plt.title('Height', fontsize=15)
        ax.xaxis.tick_top()
        ax.tick_params(axis='x', which='major', labelsize=15)
        dn = hierarchy.dendrogram(self.model,
                                  orientation='right',
                                  link_color_func=lambda k: "#000000",
                                  # truncate_mode='lastp',
                                  # p=mx_p,
                                  leaf_font_size=15,
                                  labels=labelList)
        imgstr = cvthtml.plt2html()
        print(imgstr)
        plt.clf()

        """
        ***최대 군집 수 기반 레이블 변수의 군집 할당.
        """
        # idx_list = new_idx_list.values.tolist()
        #
        # assigned_cluster = dict()
        # assigned_cluster['구분'] = idx_list
        # assigned_cluster_df = pd.DataFrame.from_dict(assigned_cluster)
        # for ik in range(self.param.k):
        #     cut_tree = fcluster(self.model, ik + 1, criterion='maxclust')
        #     k_name = '군집 개수 : {k}'.format(k=ik + 1)
        #     assigned_cluster_df.insert(ik + 1, k_name, cut_tree)

        # print(assigned_cluster_df)
        #print('<h5><b>소속군집</h5>')
        #print(assigned_cluster_df.to_html(index=False,justify='center'))
        
        # print('<h5><b>레이블 변수의 군집</b></h5>')
        cut_tree = fcluster(self.model, self.param.k, criterion='maxclust')
        cut_tree[np.where(cut_tree == 1)[0]]=0
        cut_tree[np.where(cut_tree == 2)[0]]=1

        clustered_df = pd.DataFrame({'군집번호': cut_tree, '레이블 변수': new_idx_list_train})
        # print(clustered_df.to_html(justify='center'))
        print(clustered_df)
        con_mat = pd.crosstab(clustered_df['군집번호'], clustered_df['레이블 변수'])
        y_pred = self.pred_test(data_in=self.x_val)

        # 혼동행렬
        self._plot_confusion_matrix(self.y_val.values, y_pred, False, self._CONF_MATRIX_FIT_IMG_PATH)
        # dataframe for result
        self._fit_result.y_pred = y_pred
        self._fit_result.eval_report = self._classification_report(self.y_val.values, y_pred)
        self._fit_result.confusion_matrix_img_path = self._CONF_MATRIX_FIT_IMG_PATH
        return self._fit_result

    def pred_test(self, data_in):

        adjacency_matrix_val = euclidean_distances(data_in)
        self.model = hierarchy.linkage(adjacency_matrix_val, metric='euclidean', method='complete')
        self.model[:, 2] = np.sqrt(self.model[:, 2])
        self.model[:, 2] = np.interp(self.model[:, 2], (self.model[:, 2].min(), self.model[:, 2].max()), (1, 25))

        y_pred = fcluster(self.model, self.param.k, criterion='maxclust')
        y_pred[np.where(y_pred == 1)[0]]=0
        y_pred[np.where(y_pred == 2)[0]]=1
        return y_pred

    def predict(self, target_data_path):
        """4단계: 검사데이터를 받아서 전처리 후 학습된 모델에 적용
            _preprocess: 데이터 전철
            _predict: 모델 예측

        Args:
            target_data_path (_type_): _description_

        Returns:
            _type_: _description_
        """
        self._target_data_path = target_data_path
        self.org_data_set = pd.read_csv(self._target_data_path)
        dataset = self.org_data_set.copy()
        self._preprocess(dataset, True)
        result_predict = self._predict()
        return result_predict

    def _predict(self):
        self.x_test, self.y_test = self._pred_result.x_test, self._pred_result.y_test
        self.cluster = self.model

        y_pred = self.pred_test(self.x_test)
        df_y_pred = pd.DataFrame(y_pred, columns=['예측결과'], index=self.x_test.index)
        print(df_y_pred)
        df_y_pred.columns = ['예측결과']

        if self.y_test is None:
            # y_test 값이 없을 때
            # XGboost 검사대상 Data 정보
            df_testdata = self._pred_result.x

            # 예측결과표
            df_results = self.org_data_set
            print(df_results)
            df_pred_results = df_results.join(df_y_pred)
            print(df_pred_results)


            # PredictResult 값 채우기
            self._pred_result.y_pred = y_pred
            self._pred_result.eval_report = None
            self._pred_result.confusion_matrix_img_path = None
            self._pred_result.pred_results = df_pred_results.head(100)
            self._pred_result.tf_proportion = None
        else:
            # y_test 값이 있을 때
            # XGboost 검사대상 Data 정보
            df_testdata = self._pred_result.x

            # 혼동행렬
            self._plot_confusion_matrix(self.y_test, y_pred, True, self._CONF_MATRIX_PRED_IMG_PATH)

            # 예측결과표
            df_x_results = self._pred_result.x_test
            df_y_results = self._pred_result.y_test
            df_results = df_x_results.join(df_y_results, how='inner')
            df_results['예측결과'] = y_pred
            # df_pred_results = df_results.astype({'양불판정결과': 'bool', '예측결과' : 'bool'})

            # Test Data T/F 비율
            total_count = len(df_results)
            len_true = len(df_results[df_results['예측결과'] == 1])
            len_false = len(df_results[df_results['예측결과'] == 0])
            true_proportion = len_true / len(df_results['예측결과']) * 100
            false_proportion = len_false / len(df_results['예측결과']) * 100

            data = {'총 개수(개)': [],
                    '1의 비율(%)': [],
                    '0의 비율(%)': []
                    }
            data['총 개수(개)'].append(total_count)
            data['1의 비율(%)'].append(true_proportion)
            data['0의 비율(%)'].append(false_proportion)

            df_tf_proportion = pd.DataFrame(data)
            df_tf_proportion = df_tf_proportion.round(2)

            # PredictResult 값 채우기
            self._pred_result.y_pred = y_pred
            self._pred_result.eval_report = self._classification_report(self.y_test, y_pred)
            self._pred_result.confusion_matrix_img_path = self._CONF_MATRIX_PRED_IMG_PATH
            self._pred_result.pred_results = df_results.head(100)
            self._pred_result.tf_proportion = df_tf_proportion

        return self._pred_result

    def _plot_confusion_matrix(self, y, y_pred, to_save=False, img_path='./'):

        cm = confusion_matrix(y, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues').set(title='혼동행렬')

        plt.xlabel('실제값')
        plt.ylabel('예측값')
        if to_save:
            plt.savefig(img_path)
        else:
            plt.show()

    def _classification_report(self, y, y_pred):
        df = pd.DataFrame(columns=['구분', '결과값'])
        df['구분'] = ['정확도', '정밀도', '재현율', 'f1-score']
        df['결과값'] = [accuracy_score(y, y_pred), precision_score(y, y_pred), recall_score(y, y_pred),
                     f1_score(y, y_pred)]

        return df.round(2)

'''
분석 예제:
python3 clustering_Hierarchical_v2.py --fun fit -input_path 'dataset/4.계층적군집분석_사출_unlabeled_data_1000개만.csv' -x_names 'Injection_Time','Filling_Time','Plasticizing_Time','Cycle_Time'  -target_name 'NO' -maxk 6 -mink 2 -rotate  'horizon' > output/hierrchical_injection.html
'''
if __name__ == '__main__':
    """
    계층적 클러스터링 시작.
    동작에 필요한 각종 인자를 처리한다. 
    """
    # input_path = 'dataset/kimyunashort.xlsx'
    # x_name_list = x_name_list  # ['기술점수','예술점수']
    # target_name = '심판'
    parser = argparse.ArgumentParser(prog='argparser')
    parser.add_argument('--fun', action='store_true', help='kMeans help')
    subparsers = parser.add_subparsers(help='sub-command help', dest='fun')

    # "training" 명령을 위한 파서를 만듭니다
    parser_fit = subparsers.add_parser('fit', help='fit help')
    parser_fit.add_argument('-input_path', type=str, help='input path help', required=True)
    parser_fit.add_argument('-x_names', type=str, help='x name help', required=True)
    parser_fit.add_argument('-target_name', type=str, help='target name help', required=True)
    parser_fit.add_argument('-mink', type=int, help='min of Clusters help', required=False)
    parser_fit.add_argument('-maxk', type=int, help='max of Clusters help', required=False)
    parser_fit.add_argument('-rotate', type=str, help='Chart Rotate vertical or horizon help', required=True)
    args = parser.parse_args()
    if 'fit' == args.fun:
        input_path = args.input_path
        x_name_list = args.x_names.split(',')
        target_name = args.target_name

        if args.maxk is None:
            maxk = 6
        else:
            maxk = args.maxk
        if args.mink is None:
            mink = 2
        else:
            mink = args.mink

        if args.rotate is None:
            rotate = 'vertical'
        else:
            rotate = args.rotate
        cluster = Hierachical()
        cluster.fit(input_path, x_name_list, target_name, k=maxk, rotate=rotate)