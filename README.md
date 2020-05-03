# Application of Reinforcement Learning in Laundry Sorting

# Environment:
imgaug                 0.4.0              
Keras                  2.3.1              
matplot                0.1.9              
matplotlib             3.1.1              
numpy                  1.16.0             
pandas                 0.25.3             
tensorflow             2.1.0    

# Training
- Config.py contains all the parameters for Q-Learning, SARSA, and DQN
- Environment setting on Google Colab
```python
# Environment Setting
!pip install git+https://github.com/aleju/imgaug.git
%tensorflow_version 2.x
!pip install tensorflow-gpu 
!pip install tf-nightly

# Training
!python3 LaundrySortingTrain/laundry_sorting/main.py
```


