#   pip install ucimlrepo

from ucimlrepo import fetch_ucirepo 
import pandas as pd
import matplotlib.pyplot as plt
  
# fetch dataset 
mushroom = fetch_ucirepo(id=73) 
  
# data (as pandas dataframes) 
table=mushroom.data
Features = mushroom.data.features  #caracteristicas
Objective = mushroom.data.targets  #Objective

# (a) Numero de caracteristicas
num_features=len(Features.columns)

print(f"Numero de Caracteristicas: {num_features}")

# (b) Nombre iniciales de las caracteristicas
name_Feature_List=list(Features.columns)
name_Target_List=list(Objective.columns)
counter=0
for target in name_Target_List:
    counter+=1
    print(f"Target {counter}: {target}")
counter=0
for feature in name_Feature_List:
    counter+=1
    print(f"Feature {counter}: {feature}")

#cambiamos el nombre de la feature 0 es decir del target
new_columns_target=list(Objective.columns)
new_columns_target[0]="edibility"
#modificamos el nombre de la feature 5 a oddor, sin embargo el nombre de esta feature es oddor actualmente, es decir no es necesario cambiar el nombre
Objective.columns=new_columns_target

print("Name after of changes-------------------------")

print(f"Target 1: {Objective.columns[0]}")
counter=0
for feature in name_Feature_List:
    counter+=1
    print(f"Feature {counter}: {feature}")
#(c) Modificamos los valores en la columna odor y edibility
odor_mapping = {
    'a': 'almond',
    'c': 'creosote',
    'f': 'foul',
    'l': 'anise',
    'n': 'none',
    'p': 'pungent',
    's': 'spicy',
    'y': 'fishy',
    'm': 'musty'
}
Features['odor']=Features['odor'].replace(odor_mapping)

edibility_mapping = {
    'e': 'edible',
    'p': 'poisonous'
}
Objective['edibility']=Objective['edibility'].replace(edibility_mapping)


# (d) creacion de tabla de contigencia
contingency_table=pd.crosstab(Objective['edibility'],Features['odor'])
print(contingency_table)



# Crear un histograma basado en la tabla de contingencia
contingency_table.T.plot(kind='bar', stacked=True, figsize=(10, 6))

# Configurar etiquetas y título
plt.title('Histogram of Edible and Poisonous by Odor')
plt.xlabel('Odor')
plt.ylabel('Count')
plt.legend(title='Edibility')
plt.xticks(rotation=45)
plt.grid('on')

# Mostrar el histograma
plt.tight_layout()
plt.show()


# Analizar hongos comestibles y venenosos según el olor
safe_odors = contingency_table.loc['edible'][contingency_table.loc['edible'] > 0].index.tolist()
dangerous_odors = contingency_table.loc['poisonous'][contingency_table.loc['poisonous'] > 0].index.tolist()

# Calcular probabilidades de envenenamiento por olor
probabilities = {}
for odor in Features['odor'].unique():
    edible_count = contingency_table.loc['edible', odor] if odor in contingency_table.loc['edible'] else 0
    poisonous_count = contingency_table.loc['poisonous', odor] if odor in contingency_table.loc['poisonous'] else 0
    total = edible_count + poisonous_count
    if total > 0:
        probabilities[odor] = poisonous_count / total

# Mostrar resultados
print("\nHongos seguros para comer según el olor:")
print(", ".join(safe_odors))

print("\nHongos peligrosos según el olor:")
print(", ".join(dangerous_odors))

print("\nProbabilidad de envenenamiento por olor:")
for odor, prob in probabilities.items():
    print(f"{odor}: {prob:.2%}")
