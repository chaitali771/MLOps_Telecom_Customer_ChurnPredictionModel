from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import MinMaxScaler,OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer,KNNImputer
from sklearn.compose import ColumnTransformer
from imblearn.over_sampling import SMOTE




def data_preprocessing(df):

    '''
    * step1 : Drop the duplicate values present in the dataset.

    * step2 : Understand descriptive stats abt the dataset and use feature engineering (Optional)**

    * step3 : Split the dataset into independent and dependent columns i.e X and y.

    * step4 : Split the independent and dependent columns into train(Seen) and test(Test) dataset.

    * step5 : Segreggate the pipeline into numerical_pipeline and categorical_pipeline.

    * step6 : Use imputation and Scaling Technique on numerical_pipeline

    * step7: Use Imputation and Encoding Technique on categorical_pipeline

    * step8 : Use columntransformer to concat numerical_pipeline and categorical_pipeline

    * step9 : Fit the transformer on training and testing (On Seen and Unseen dataset).

    * step10 :fit_resample the data so that target column balance  (in the case of classification)
       
    *step11 : Return the preprocessed X_tarin,X_tset,y_train,y_test 
    
    '''
   
    # STEP 1 : Ensuring drop the duplicate values.
    df.drop_duplicates(inplace = True)

    # Step 2 : Feature Engineering (Optional).

    # Step 3 : Split the dataset into  X and y.

    X = df.drop(columns = ['customerID','Churn'])
    y = df['Churn']                   # Dependent column

    # Step 4 : Split the dataset into train(Seen) and test(Test) dataset.

    X_train, X_test, y_train, y_test = train_test_split(X,y,
                                                        test_size = 0.3,
                                                        random_state = 1)

    # Step 5 : Segregate the pipeline into numerical and categorical.

        # Segregate the numerical and categorical columns.
    numerical_col = X_train.select_dtypes(exclude = 'object').columns
    categorical_col = X_train.select_dtypes(include = 'object').columns

    # Step 6 : Use numerical_pipeline

    numerical_pipeline = Pipeline([('Imputation',SimpleImputer(strategy = 'median')),
                                   ('Scaling',MinMaxScaler())])
    # Step 7 : Use categorical_pipeline
    
    categorical_pipeline = Pipeline([('Imputation',SimpleImputer(strategy = 'most_frequent')),
                                     ('Encoding',OneHotEncoder(handle_unknown = 'ignore'))])

    # Step 8 : Use columntransformer to concat numerical_pipeline and categorical_pipeline
     
    transformer = ColumnTransformer([('numerical_pipeline',
                                        numerical_pipeline,
                                        numerical_col),
                                        ('categorical_pipeline',
                                        categorical_pipeline,
                                        categorical_col)])

    # Step 9 : Fit the transformer on seen and unseen dataset.

    X_train = transformer.fit_transform(X_train)   #Seen dataset
    X_test = transformer.transform(X_test)         #Unseen dataset
    
    # Step 10 :
    # fit_resample the data so that target column balance
    '''
    generally used for classification model
    target column balance technique basically classified into two....
    - over sampling technique
    - under sampling technique
    
    '''
    X_train,y_train = SMOTE().fit_resample(X_train,y_train)

    # Step 11 : Return the preprocess data value

    return X_train, X_test, y_train, y_test
    
