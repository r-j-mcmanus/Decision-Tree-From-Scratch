import pandas as pd
import numpy as np

MIN_TO_SPLIT = 1
MAX_DEPTH = 3
MIN_GINI_GAIN = 0.001

class DT_node():
	'''
	A decicion tree node, when called by the user will act as the root node. 
	'''
	def __init__(self, df, y_label, depth = 0):
		'''
		When called by the user will act as the root for a decision tree.

		Will be trained on the dataframe df with target variables label y_label.

		The depth parameter does not need to be changed.

		Arguments
		---------
		df : pandas dataframe containing the target variables and feature matrix
		y_label : the colomn label for the target variable in the dataframe
		depth : the depth of the current node, used in recursive calls, does not need to be changed
		'''
		self.gini_Score = self.calc_gini(df[y_label])
		self.child_left = None
		self.child_right = None

		depth = depth + 1

		if df.shape[0] <= MIN_TO_SPLIT:
			self.make_leaf_features(df, y_label)
			return
		if depth > MAX_DEPTH:
			self.make_leaf_features(df, y_label)
			return

		else:
			self.best_feature, self.best_split, self.gini_gain = self.find_best_split(y_label, df)
			
			if self.gini_gain <= 0:
				self.make_leaf_features(df, y_label)
				return

			if self.gini_gain < MIN_GINI_GAIN:
				self.make_leaf_features(df, y_label)
				return

			self.child_left = DT_node(df[df[self.best_feature] < self.best_split], y_label, depth)
			self.child_right = DT_node(df[df[self.best_feature] >= self.best_split], y_label, depth)

	def make_leaf_features(self, df, y_label):
		'''
		when the node is a leaf, it will add the instance variables prediction and node_sample_density. 

		prediction is the mode of the training data in the node
		node_sample_density is the % of the data equal to the mode
		'''
			self.prediction = df[y_label].mode()[0]
			self.node_sample_density = (df[y_label].values == self.prediction).sum()

	def calc_gini(self, y):
		'''
		Calculates the gini impurity of the target variable y
		'''
		# as we are incrementing y by one each time we could optimize this rather than call value count each time
		value_counts = y.value_counts(normalize = True)
		gini = 1 - value_counts.apply(lambda x : x*x ).sum()
		return gini

	def find_best_split(self, y_label, df):
		'''
		Will find a value for any feature collumn in the data frame which maximises the gini purity gain
		'''
		max_gini_gain = 0
		best_feature = None
		best_split = None

		for col in df.columns:
			if col == y_label:
				continue

			sdf = df.dropna().sort_values(col)
			col_len = sdf[col].size

			for i in range(col_len):
				gini_left = self.calc_gini(sdf[y_label].iloc[:i])
				gini_right = self.calc_gini(sdf[y_label].iloc[i:])

				#weights for the above gini values
				w_left = (i+1) / col_len
				w_right =  (col_len - i - 1) / col_len

				# the weighted sum of gini values
				gini_weighted = w_left * gini_left + w_right * gini_right
				
				gini_gain = self.gini_Score - gini_weighted

				if(gini_gain > max_gini_gain):
					max_gini_gain = gini_gain
					best_feature = col
					best_split = sdf[col].iloc[i]

		return best_feature, best_split, max_gini_gain

	def predict(self, X_test):
		'''
		When called by the user, will predict values of the test data X_test

		Arguments
		---------
		X_test : a feature matrix in the form of a pandas dataframe

		'''
		y_pred = pd.Series(index = X_test.index)
		mask = pd.Series(data = [True for i in range(X_test.shape[0])], index = X_test.index)
		self.sub_predict(X_test, y_pred, mask)
		return y_pred

	def sub_predict(self, X_test, y_pred, mask):
		'''
		Called recusivly in the tree.

		If node has no children, applys the mask to the y_pred and sets the predicted value 
		'''
		if self.child_left != None:
			self.hand_to_children(X_test, y_pred, mask)
		else:
			y_pred[mask] = self.prediction

	def hand_to_children(self,X_test, y_pred, mask):
		'''
		makes a mask and hands to the children nodes
		'''
		mask_left = mask & (X_test[self.best_feature] < self.best_split )
		self.child_left.sub_predict(X_test, y_pred, mask_left)

		mask_right = mask & (X_test[self.best_feature] >= self.best_split)
		self.child_right.sub_predict(X_test, y_pred, mask_right)


def Make_DF(zeroes_number = 100, ones_number = 100):
	y = np.concatenate((np.zeros(zeroes_number), np.ones(ones_number)))

	x1 = np.concatenate((np.random.rand(zeroes_number),
			np.random.rand(ones_number)+0.5))

	x2 = np.concatenate((np.random.rand(zeroes_number),
			np.random.rand(ones_number)+0.5))

	x3 = np.concatenate((np.random.rand(zeroes_number),
			np.random.rand(ones_number)+0.5))

	my_dict = {'y':y,'x1':x1,'x2':x2, 'x3':x3}

	return pd.DataFrame(my_dict)

df_explicit = Make_DF(200,200)
df_tree = DT_node(df_explicit, 'y')

df_explicit = Make_DF(0,200)
y_pred = df_tree.predict(df_explicit.drop('y', axis = 1))

print(y_pred.value_counts())