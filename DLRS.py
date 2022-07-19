#collaborative filtering
#different users and read different items and tring to rpedict missing ratings for every user. 
#one for the user and one for the items.

import pandas as pd
from sklearn import model_selection
import torch
import torch.nn as nn
from sklearn import metrics, preprocessing
import numpy as np
#split the data set into training and testing 
df = pd.read_csv("input/train_v2.csv")
df.user.nunique()
df.movie.nunique()
df.shape
df.rating.value_counts()



class MovieDataset:
 def __init__(self, users, movies, ratings):
  self.users = users
  self.movies = movies
  self.ratings = ratings
 def __len__(self):
  return len(self.users)
 def __getitem__(self, item):
  user = self.users[item]
  movies = self.movies[item]
  rating = self.ratings[item]
  return {"user": torch.tensor(user, dtype=torch.long),
          "movie": torch.tensor(movie, dtype=torch.long), 
           "rating": torch.tensor(rating, dtype=torch.float)
          }

class RecSysModel(tez.Model):
 def __init__(self, num_users, num_movies):
   super().__init__()
   self.user_embed = nn.Embedding(num_users, 32)
   self.movie_embed = nn.Embedding(num_movies, 32)
   self.out = nn.Linear(64, 1)
   self.step_scheduler_after = "epoch"

 def fetch_optimizer(self):
   opt = torch.optim.Adam(self.parameters(), lr=1e-3)
   return opt
 def fetch_scheduler(self):
   sch = torch.optim.lr_scheduler.StepLP(self.optimizer, step_size=3, gamma=0.7)
   return opt
 def forward(self, user, movies, rating): 
   user_embeds = self.user_embed(users) 
   movie_embed = self.movie_embed(movies) output = torch.cat([user_embeds, movie_embeds], dim=1) 
   output = self.out(output)

    #we can monitor RMSE most widely.
   if ratings:
      loss = nnn.MSE(output, ratings.view(-1,1))
      calc_metrics = self.monitor_metrics(output, ratings.view(-1, 1))
      return output, loss, calc_metrics

  def monitor_mterics(self, output, rating):
    output = output.detach().cpu().numpy()
    rating = rating.detach().cpu().numpy()
     return {
        'rmse': np.sqrt(metrics.mean_squared_erro(rating, output))
     }


def train():
 df = pd.read_csv("../input/train_v2.csv")

 lbl_user = preprocessing.LabelEncoder()
 lbl_movie = preprocessing.LabelEncoder()
 df.user = lbl_user.fit_transform(df.user.values)
 df.movie = lbl_movie.fit_transform(df.movie.values)
 #I have got four columns, id, user, movie, rating
 df_train, df_valid = model_selection.train_test_split(df, test_size = 0.1, random_State=42, stratify= df.rating.values)
 #keep the ratio as it is. Now, we will testify our testing data set 
 train_dataset = MovieDataset(users = df_train.user.values,movies =  df_train.movie.values,ratings = df_train.ratings.values)
 valid_dataset = MovieDataset(
  users = valid.user.values, movies = df_valid.movie.values, ratings = df_valid.ratings.values
 )

 model = RecSysModel(num_users = len(lbl_user.classes_), num_movies = len(lbl_movie.classes_))
 model.fit(
        train_dataset, valid_dataset, train_bs = 1024, valid_bs = 1024, fp16 = True
 )

if __name__=="__main__":
 train()
