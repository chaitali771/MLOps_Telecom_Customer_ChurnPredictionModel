from src.data_ingestion import data_ingestion
from src.data_preprocessing import data_preprocessing
from src.model_building import model_build
from src.model_prediction import model_predict
from src.model_evaluation import model_evaluate
def main():
    # Step1: Data Ingestion
    df = data_ingestion()
    print(df.columns)
    
    # step2: Data Preprocessing
    X_train_resample, X_test_transform, y_train_resample, y_test= data_preprocessing(df)
    
    #step3: model building
    model = model_build(X_train_resample,X_test_transform,y_train_resample,y_test)
    
    #step4: model prediction 
    y_pred = model_predict(model , X_test_transform)
    
    #step5: model_evaluation
    score = model_evaluate(y_test,y_pred)
    print(f'model classification report: {score}')
main()    