
# requirements
Numpy
Sklearn 
mlxtend
xgboost, optional

# fichiers 
Il y a trois fichiers :
- read_data.py
- utils.py
- predict.py
le premier permet la lecture et l'écriture.
le second fourni quelques fonction notament pour la notation.
le troiseme est le principal avec touts les modèles.

# Lancer les modèles 
le fichier à executer es prédict.py en mode interactif
   python -i predict.py

Plusieurs modèless on été préconstruits avec leur hyperparametres sélectionnés.
pour en construire un il suffit de : 

    mod = modele()

 avec modele parmis 

pour instancier un modèle parmis :
- Svc
- MLP
- MLP2
- RandomForest
- GradientBoost
- Knlambda
- logistic
- Stacking
- simple_stacing
- Voting

pour le tester en local il suffit de lancer
    
    train_test(mod)

pour constuire une solution kaggle :
    
    train_fullset(mod, filename)

Il créera un fichier nomé filename dans le répretoire courant

pour ceci il faut que __test.csv__ et __train.csv__ soient dans le répertoire parent.

