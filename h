{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# [Problem 1] Data acquisition"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.datasets import load_iris\n",
    "iris= load_iris()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>sepal length (cm)</th>\n",
       "      <th>sepal width (cm)</th>\n",
       "      <th>petal length (cm)</th>\n",
       "      <th>petal width (cm)</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>5.1</td>\n",
       "      <td>3.5</td>\n",
       "      <td>1.4</td>\n",
       "      <td>0.2</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>4.9</td>\n",
       "      <td>3.0</td>\n",
       "      <td>1.4</td>\n",
       "      <td>0.2</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>4.7</td>\n",
       "      <td>3.2</td>\n",
       "      <td>1.3</td>\n",
       "      <td>0.2</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>4.6</td>\n",
       "      <td>3.1</td>\n",
       "      <td>1.5</td>\n",
       "      <td>0.2</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>5.0</td>\n",
       "      <td>3.6</td>\n",
       "      <td>1.4</td>\n",
       "      <td>0.2</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>...</th>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>145</th>\n",
       "      <td>6.7</td>\n",
       "      <td>3.0</td>\n",
       "      <td>5.2</td>\n",
       "      <td>2.3</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>146</th>\n",
       "      <td>6.3</td>\n",
       "      <td>2.5</td>\n",
       "      <td>5.0</td>\n",
       "      <td>1.9</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>147</th>\n",
       "      <td>6.5</td>\n",
       "      <td>3.0</td>\n",
       "      <td>5.2</td>\n",
       "      <td>2.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>148</th>\n",
       "      <td>6.2</td>\n",
       "      <td>3.4</td>\n",
       "      <td>5.4</td>\n",
       "      <td>2.3</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>149</th>\n",
       "      <td>5.9</td>\n",
       "      <td>3.0</td>\n",
       "      <td>5.1</td>\n",
       "      <td>1.8</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>150 rows × 4 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "     sepal length (cm)  sepal width (cm)  petal length (cm)  petal width (cm)\n",
       "0                  5.1               3.5                1.4               0.2\n",
       "1                  4.9               3.0                1.4               0.2\n",
       "2                  4.7               3.2                1.3               0.2\n",
       "3                  4.6               3.1                1.5               0.2\n",
       "4                  5.0               3.6                1.4               0.2\n",
       "..                 ...               ...                ...               ...\n",
       "145                6.7               3.0                5.2               2.3\n",
       "146                6.3               2.5                5.0               1.9\n",
       "147                6.5               3.0                5.2               2.0\n",
       "148                6.2               3.4                5.4               2.3\n",
       "149                5.9               3.0                5.1               1.8\n",
       "\n",
       "[150 rows x 4 columns]"
      ]
     },
     "execution_count": 2,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "import pandas as pd\n",
    "import numpy as np\n",
    "\n",
    "# Store features matrix in X\n",
    "X = pd.DataFrame(data= np.c_[iris['data']],\n",
    "                  columns= iris['feature_names'])\n",
    "X"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Species</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>...</th>\n",
       "      <td>...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>145</th>\n",
       "      <td>2</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>146</th>\n",
       "      <td>2</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>147</th>\n",
       "      <td>2</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>148</th>\n",
       "      <td>2</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>149</th>\n",
       "      <td>2</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>150 rows × 1 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "     Species\n",
       "0          0\n",
       "1          0\n",
       "2          0\n",
       "3          0\n",
       "4          0\n",
       "..       ...\n",
       "145        2\n",
       "146        2\n",
       "147        2\n",
       "148        2\n",
       "149        2\n",
       "\n",
       "[150 rows x 1 columns]"
      ]
     },
     "execution_count": 3,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#Store target vector in\n",
    "y =  pd.DataFrame(data= np.c_[iris['target']],\n",
    "                  columns=['Species'])\n",
    "y"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# [Problem 2] Combining data"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>sepal length (cm)</th>\n",
       "      <th>sepal width (cm)</th>\n",
       "      <th>petal length (cm)</th>\n",
       "      <th>petal width (cm)</th>\n",
       "      <th>Species</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>5.1</td>\n",
       "      <td>3.5</td>\n",
       "      <td>1.4</td>\n",
       "      <td>0.2</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>4.9</td>\n",
       "      <td>3.0</td>\n",
       "      <td>1.4</td>\n",
       "      <td>0.2</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>4.7</td>\n",
       "      <td>3.2</td>\n",
       "      <td>1.3</td>\n",
       "      <td>0.2</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>4.6</td>\n",
       "      <td>3.1</td>\n",
       "      <td>1.5</td>\n",
       "      <td>0.2</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>5.0</td>\n",
       "      <td>3.6</td>\n",
       "      <td>1.4</td>\n",
       "      <td>0.2</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>...</th>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>145</th>\n",
       "      <td>6.7</td>\n",
       "      <td>3.0</td>\n",
       "      <td>5.2</td>\n",
       "      <td>2.3</td>\n",
       "      <td>2</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>146</th>\n",
       "      <td>6.3</td>\n",
       "      <td>2.5</td>\n",
       "      <td>5.0</td>\n",
       "      <td>1.9</td>\n",
       "      <td>2</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>147</th>\n",
       "      <td>6.5</td>\n",
       "      <td>3.0</td>\n",
       "      <td>5.2</td>\n",
       "      <td>2.0</td>\n",
       "      <td>2</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>148</th>\n",
       "      <td>6.2</td>\n",
       "      <td>3.4</td>\n",
       "      <td>5.4</td>\n",
       "      <td>2.3</td>\n",
       "      <td>2</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>149</th>\n",
       "      <td>5.9</td>\n",
       "      <td>3.0</td>\n",
       "      <td>5.1</td>\n",
       "      <td>1.8</td>\n",
       "      <td>2</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>150 rows × 5 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "     sepal length (cm)  sepal width (cm)  petal length (cm)  petal width (cm)  \\\n",
       "0                  5.1               3.5                1.4               0.2   \n",
       "1                  4.9               3.0                1.4               0.2   \n",
       "2                  4.7               3.2                1.3               0.2   \n",
       "3                  4.6               3.1                1.5               0.2   \n",
       "4                  5.0               3.6                1.4               0.2   \n",
       "..                 ...               ...                ...               ...   \n",
       "145                6.7               3.0                5.2               2.3   \n",
       "146                6.3               2.5                5.0               1.9   \n",
       "147                6.5               3.0                5.2               2.0   \n",
       "148                6.2               3.4                5.4               2.3   \n",
       "149                5.9               3.0                5.1               1.8   \n",
       "\n",
       "     Species  \n",
       "0          0  \n",
       "1          0  \n",
       "2          0  \n",
       "3          0  \n",
       "4          0  \n",
       "..       ...  \n",
       "145        2  \n",
       "146        2  \n",
       "147        2  \n",
       "148        2  \n",
       "149        2  \n",
       "\n",
       "[150 rows x 5 columns]"
      ]
     },
     "execution_count": 4,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df = pd.concat([X, y], axis=1)\n",
    "df"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# [Problem 3] Checking the data"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>sepal length (cm)</th>\n",
       "      <th>sepal width (cm)</th>\n",
       "      <th>petal length (cm)</th>\n",
       "      <th>petal width (cm)</th>\n",
       "      <th>Species</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>5.1</td>\n",
       "      <td>3.5</td>\n",
       "      <td>1.4</td>\n",
       "      <td>0.2</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>4.9</td>\n",
       "      <td>3.0</td>\n",
       "      <td>1.4</td>\n",
       "      <td>0.2</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>4.7</td>\n",
       "      <td>3.2</td>\n",
       "      <td>1.3</td>\n",
       "      <td>0.2</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>4.6</td>\n",
       "      <td>3.1</td>\n",
       "      <td>1.5</td>\n",
       "      <td>0.2</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>5.0</td>\n",
       "      <td>3.6</td>\n",
       "      <td>1.4</td>\n",
       "      <td>0.2</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   sepal length (cm)  sepal width (cm)  petal length (cm)  petal width (cm)  \\\n",
       "0                5.1               3.5                1.4               0.2   \n",
       "1                4.9               3.0                1.4               0.2   \n",
       "2                4.7               3.2                1.3               0.2   \n",
       "3                4.6               3.1                1.5               0.2   \n",
       "4                5.0               3.6                1.4               0.2   \n",
       "\n",
       "   Species  \n",
       "0        0  \n",
       "1        0  \n",
       "2        0  \n",
       "3        0  \n",
       "4        0  "
      ]
     },
     "execution_count": 5,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#Display the 4th sample from the beginning and see what each feature is like\n",
    "df.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>sepal length (cm)</th>\n",
       "      <th>sepal width (cm)</th>\n",
       "      <th>petal length (cm)</th>\n",
       "      <th>petal width (cm)</th>\n",
       "      <th>Species</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>5.1</td>\n",
       "      <td>3.5</td>\n",
       "      <td>1.4</td>\n",
       "      <td>0.2</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>4.9</td>\n",
       "      <td>3.0</td>\n",
       "      <td>1.4</td>\n",
       "      <td>0.2</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>4.7</td>\n",
       "      <td>3.2</td>\n",
       "      <td>1.3</td>\n",
       "      <td>0.2</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>4.6</td>\n",
       "      <td>3.1</td>\n",
       "      <td>1.5</td>\n",
       "      <td>0.2</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>5.0</td>\n",
       "      <td>3.6</td>\n",
       "      <td>1.4</td>\n",
       "      <td>0.2</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   sepal length (cm)  sepal width (cm)  petal length (cm)  petal width (cm)  \\\n",
       "0                5.1               3.5                1.4               0.2   \n",
       "1                4.9               3.0                1.4               0.2   \n",
       "2                4.7               3.2                1.3               0.2   \n",
       "3                4.6               3.1                1.5               0.2   \n",
       "4                5.0               3.6                1.4               0.2   \n",
       "\n",
       "   Species  \n",
       "0        0  \n",
       "1        0  \n",
       "2        0  \n",
       "3        0  \n",
       "4        0  "
      ]
     },
     "execution_count": 6,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "<class 'pandas.core.frame.DataFrame'>\n",
      "RangeIndex: 150 entries, 0 to 149\n",
      "Data columns (total 5 columns):\n",
      " #   Column             Non-Null Count  Dtype  \n",
      "---  ------             --------------  -----  \n",
      " 0   sepal length (cm)  150 non-null    float64\n",
      " 1   sepal width (cm)   150 non-null    float64\n",
      " 2   petal length (cm)  150 non-null    float64\n",
      " 3   petal width (cm)   150 non-null    float64\n",
      " 4   Species            150 non-null    int32  \n",
      "dtypes: float64(4), int32(1)\n",
      "memory usage: 5.4 KB\n"
     ]
    }
   ],
   "source": [
    "df.info()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>sepal length (cm)</th>\n",
       "      <th>sepal width (cm)</th>\n",
       "      <th>petal length (cm)</th>\n",
       "      <th>petal width (cm)</th>\n",
       "      <th>Species</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>count</th>\n",
       "      <td>150.000000</td>\n",
       "      <td>150.000000</td>\n",
       "      <td>150.000000</td>\n",
       "      <td>150.000000</td>\n",
       "      <td>150.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>mean</th>\n",
       "      <td>5.843333</td>\n",
       "      <td>3.057333</td>\n",
       "      <td>3.758000</td>\n",
       "      <td>1.199333</td>\n",
       "      <td>1.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>std</th>\n",
       "      <td>0.828066</td>\n",
       "      <td>0.435866</td>\n",
       "      <td>1.765298</td>\n",
       "      <td>0.762238</td>\n",
       "      <td>0.819232</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>min</th>\n",
       "      <td>4.300000</td>\n",
       "      <td>2.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>0.100000</td>\n",
       "      <td>0.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>25%</th>\n",
       "      <td>5.100000</td>\n",
       "      <td>2.800000</td>\n",
       "      <td>1.600000</td>\n",
       "      <td>0.300000</td>\n",
       "      <td>0.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>50%</th>\n",
       "      <td>5.800000</td>\n",
       "      <td>3.000000</td>\n",
       "      <td>4.350000</td>\n",
       "      <td>1.300000</td>\n",
       "      <td>1.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>75%</th>\n",
       "      <td>6.400000</td>\n",
       "      <td>3.300000</td>\n",
       "      <td>5.100000</td>\n",
       "      <td>1.800000</td>\n",
       "      <td>2.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>max</th>\n",
       "      <td>7.900000</td>\n",
       "      <td>4.400000</td>\n",
       "      <td>6.900000</td>\n",
       "      <td>2.500000</td>\n",
       "      <td>2.000000</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "       sepal length (cm)  sepal width (cm)  petal length (cm)  \\\n",
       "count         150.000000        150.000000         150.000000   \n",
       "mean            5.843333          3.057333           3.758000   \n",
       "std             0.828066          0.435866           1.765298   \n",
       "min             4.300000          2.000000           1.000000   \n",
       "25%             5.100000          2.800000           1.600000   \n",
       "50%             5.800000          3.000000           4.350000   \n",
       "75%             6.400000          3.300000           5.100000   \n",
       "max             7.900000          4.400000           6.900000   \n",
       "\n",
       "       petal width (cm)     Species  \n",
       "count        150.000000  150.000000  \n",
       "mean           1.199333    1.000000  \n",
       "std            0.762238    0.819232  \n",
       "min            0.100000    0.000000  \n",
       "25%            0.300000    0.000000  \n",
       "50%            1.300000    1.000000  \n",
       "75%            1.800000    2.000000  \n",
       "max            2.500000    2.000000  "
      ]
     },
     "execution_count": 8,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#Output the total number of samples for each label\n",
    "df.describe()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "sepal length (cm)    False\n",
       "sepal width (cm)     False\n",
       "petal length (cm)    False\n",
       "petal width (cm)     False\n",
       "Species              False\n",
       "dtype: bool"
      ]
     },
     "execution_count": 9,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#Check if there is a missing value in the feature quantity\n",
    "df.isnull().any()#no missing values"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "sepal length (cm)    876.5\n",
       "sepal width (cm)     458.6\n",
       "petal length (cm)    563.7\n",
       "petal width (cm)     179.9\n",
       "Species              150.0\n",
       "dtype: float64"
      ]
     },
     "execution_count": 10,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.sum()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "sepal length (cm)  sepal width (cm)  petal length (cm)  petal width (cm)  Species\n",
       "5.8                2.7               5.1                1.9               2          2\n",
       "7.9                3.8               6.4                2.0               2          1\n",
       "5.4                3.0               4.5                1.5               1          1\n",
       "5.5                2.4               3.7                1.0               1          1\n",
       "                   2.3               4.0                1.3               1          1\n",
       "                                                                                    ..\n",
       "6.3                2.5               4.9                1.5               1          1\n",
       "                   2.3               4.4                1.3               1          1\n",
       "6.2                3.4               5.4                2.3               2          1\n",
       "                   2.9               4.3                1.3               1          1\n",
       "4.3                3.0               1.1                0.1               0          1\n",
       "Length: 149, dtype: int64"
      ]
     },
     "execution_count": 11,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.value_counts()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## [Problem 4] Examining the iris dataset itself"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "- There are 150 observations with 4 features each (sepal length, sepal width, petal length, petal width).\n",
    "- There are no null values.\n",
    "- There are 50 observations of each species (setosa, versicolor, virginica).\n",
    "- Each row is an observation (also known as: sample, example, instance, record)\n",
    "- Each column is a feature (also known as: predictor, attribute, independent variable, input, regressor)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0      0\n",
       "1      0\n",
       "2      0\n",
       "3      0\n",
       "4      0\n",
       "      ..\n",
       "145    2\n",
       "146    2\n",
       "147    2\n",
       "148    2\n",
       "149    2\n",
       "Name: Species, Length: 150, dtype: int32"
      ]
     },
     "execution_count": 12,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.Species"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array(['setosa', 'versicolor', 'virginica'], dtype='<U10')"
      ]
     },
     "execution_count": 13,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "iris.target_names"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "- \"0\": setosa\n",
    "- \"1\": versicolor\n",
    "- \"2\": virginica"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "2    50\n",
       "1    50\n",
       "0    50\n",
       "Name: Species, dtype: int64"
      ]
     },
     "execution_count": 14,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df['Species'].value_counts()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## [Problem 5] Extracting the required data"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Extract sepal_width, columns in three different ways\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 46,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>sepal width (cm)</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>3.5</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>3.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>3.2</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>3.1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>3.6</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   sepal width (cm)\n",
       "0               3.5\n",
       "1               3.0\n",
       "2               3.2\n",
       "3               3.1\n",
       "4               3.6"
      ]
     },
     "execution_count": 46,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# method 1 using iloc()\n",
    "df.iloc[:,1:2].head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>sepal width (cm)</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>3.5</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>3.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>3.2</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>3.1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>3.6</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   sepal width (cm)\n",
       "0               3.5\n",
       "1               3.0\n",
       "2               3.2\n",
       "3               3.1\n",
       "4               3.6"
      ]
     },
     "execution_count": 16,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# METHOD 2 USING LOC()\n",
    "df.loc[:,['sepal width (cm)']].head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>sepal width (cm)</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>3.5</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>3.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>3.2</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>3.1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>3.6</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   sepal width (cm)\n",
       "0               3.5\n",
       "1               3.0\n",
       "2               3.2\n",
       "3               3.1\n",
       "4               3.6"
      ]
     },
     "execution_count": 17,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# method 3 using local index position \n",
    "\n",
    "df[['sepal width (cm)']].head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>sepal length (cm)</th>\n",
       "      <th>sepal width (cm)</th>\n",
       "      <th>petal length (cm)</th>\n",
       "      <th>petal width (cm)</th>\n",
       "      <th>Species</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>50</th>\n",
       "      <td>7.0</td>\n",
       "      <td>3.2</td>\n",
       "      <td>4.7</td>\n",
       "      <td>1.4</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>51</th>\n",
       "      <td>6.4</td>\n",
       "      <td>3.2</td>\n",
       "      <td>4.5</td>\n",
       "      <td>1.5</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>52</th>\n",
       "      <td>6.9</td>\n",
       "      <td>3.1</td>\n",
       "      <td>4.9</td>\n",
       "      <td>1.5</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>53</th>\n",
       "      <td>5.5</td>\n",
       "      <td>2.3</td>\n",
       "      <td>4.0</td>\n",
       "      <td>1.3</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>54</th>\n",
       "      <td>6.5</td>\n",
       "      <td>2.8</td>\n",
       "      <td>4.6</td>\n",
       "      <td>1.5</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "    sepal length (cm)  sepal width (cm)  petal length (cm)  petal width (cm)  \\\n",
       "50                7.0               3.2                4.7               1.4   \n",
       "51                6.4               3.2                4.5               1.5   \n",
       "52                6.9               3.1                4.9               1.5   \n",
       "53                5.5               2.3                4.0               1.3   \n",
       "54                6.5               2.8                4.6               1.5   \n",
       "\n",
       "    Species  \n",
       "50        1  \n",
       "51        1  \n",
       "52        1  \n",
       "53        1  \n",
       "54        1  "
      ]
     },
     "execution_count": 18,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Extract the 50th to 99th data\n",
    "df.iloc[50:100, :]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>petal length (cm)</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>50</th>\n",
       "      <td>4.7</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>51</th>\n",
       "      <td>4.5</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>52</th>\n",
       "      <td>4.9</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>53</th>\n",
       "      <td>4.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>54</th>\n",
       "      <td>4.6</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>55</th>\n",
       "      <td>4.5</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>56</th>\n",
       "      <td>4.7</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>57</th>\n",
       "      <td>3.3</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>58</th>\n",
       "      <td>4.6</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>59</th>\n",
       "      <td>3.9</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>60</th>\n",
       "      <td>3.5</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>61</th>\n",
       "      <td>4.2</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>62</th>\n",
       "      <td>4.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>63</th>\n",
       "      <td>4.7</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>64</th>\n",
       "      <td>3.6</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>65</th>\n",
       "      <td>4.4</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>66</th>\n",
       "      <td>4.5</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>67</th>\n",
       "      <td>4.1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>68</th>\n",
       "      <td>4.5</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>69</th>\n",
       "      <td>3.9</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "    petal length (cm)\n",
       "50                4.7\n",
       "51                4.5\n",
       "52                4.9\n",
       "53                4.0\n",
       "54                4.6\n",
       "55                4.5\n",
       "56                4.7\n",
       "57                3.3\n",
       "58                4.6\n",
       "59                3.9\n",
       "60                3.5\n",
       "61                4.2\n",
       "62                4.0\n",
       "63                4.7\n",
       "64                3.6\n",
       "65                4.4\n",
       "66                4.5\n",
       "67                4.1\n",
       "68                4.5\n",
       "69                3.9"
      ]
     },
     "execution_count": 19,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Extract the50th to 99th data of the petal_length column\n",
    "df.loc[50:99, ['petal length (cm)']]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>sepal length (cm)</th>\n",
       "      <th>sepal width (cm)</th>\n",
       "      <th>petal length (cm)</th>\n",
       "      <th>petal width (cm)</th>\n",
       "      <th>Species</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>5.1</td>\n",
       "      <td>3.5</td>\n",
       "      <td>1.4</td>\n",
       "      <td>0.2</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>4.9</td>\n",
       "      <td>3.0</td>\n",
       "      <td>1.4</td>\n",
       "      <td>0.2</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>4.7</td>\n",
       "      <td>3.2</td>\n",
       "      <td>1.3</td>\n",
       "      <td>0.2</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>4.6</td>\n",
       "      <td>3.1</td>\n",
       "      <td>1.5</td>\n",
       "      <td>0.2</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>5.0</td>\n",
       "      <td>3.6</td>\n",
       "      <td>1.4</td>\n",
       "      <td>0.2</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   sepal length (cm)  sepal width (cm)  petal length (cm)  petal width (cm)  \\\n",
       "0                5.1               3.5                1.4               0.2   \n",
       "1                4.9               3.0                1.4               0.2   \n",
       "2                4.7               3.2                1.3               0.2   \n",
       "3                4.6               3.1                1.5               0.2   \n",
       "4                5.0               3.6                1.4               0.2   \n",
       "\n",
       "   Species  \n",
       "0        0  \n",
       "1        0  \n",
       "2        0  \n",
       "3        0  \n",
       "4        0  "
      ]
     },
     "execution_count": 20,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#Extract data with a petal_width value of 0.2\n",
    "df[df['petal width (cm)'] == 0.2].head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>petal width (cm)</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>0.2</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>0.2</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>0.2</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>0.2</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>0.2</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   petal width (cm)\n",
       "0               0.2\n",
       "1               0.2\n",
       "2               0.2\n",
       "3               0.2\n",
       "4               0.2"
      ]
     },
     "execution_count": 21,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.loc[df['petal width (cm)'] == 0.2,['petal width (cm)']].head()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# [Problem 6] Creating a diagram"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "import matplotlib.pyplot as plt\n",
    "%matplotlib inline\n",
    "\n",
    "a = df['Species'].value_counts()\n",
    "species = a.index\n",
    "count = a.values\n",
    "colors = ['blue', 'green', 'purple']\n",
    "#explore = (0,0.2,0)\n",
    "plt.pie(count, labels=species,colors = colors, autopct = '%1.1f%%')\n",
    "plt.xlabel('species')\n",
    "plt.axis('equal')\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<AxesSubplot:>"
      ]
     },
     "execution_count": 23,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "# visualizing the distribution of data for each label using a box plot\n",
    "df.boxplot()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<AxesSubplot:>"
      ]
     },
     "execution_count": 24,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "df.boxplot(['sepal length (cm)'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<AxesSubplot:>"
      ]
     },
     "execution_count": 25,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "df.boxplot(['sepal width (cm)'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<AxesSubplot:>"
      ]
     },
     "execution_count": 26,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "df.boxplot(['petal length (cm)'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 27,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<AxesSubplot:>"
      ]
     },
     "execution_count": 27,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "df.boxplot(['petal width (cm)'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 28,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<AxesSubplot:ylabel='sepal length (cm)'>"
      ]
     },
     "execution_count": 28,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 360x216 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "# visualize the distribution of data for each label using a violin plot\n",
    "import seaborn as sns\n",
    "fig, ax = plt.subplots(figsize =(5, 3)) \n",
    "sns.violinplot( ax = ax, y = df[\"sepal length (cm)\"] ) \n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 29,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<AxesSubplot:ylabel='sepal width (cm)'>"
      ]
     },
     "execution_count": 29,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 360x216 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "fig, ax = plt.subplots(figsize =(5, 3)) \n",
    "sns.violinplot( ax = ax, y = df[\"sepal width (cm)\"] ) \n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 30,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<AxesSubplot:ylabel='petal length (cm)'>"
      ]
     },
     "execution_count": 30,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAAUAAAAC0CAYAAADl5PURAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjMuMiwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy8vihELAAAACXBIWXMAAAsTAAALEwEAmpwYAAAt8UlEQVR4nO3deVzU95348ddnDq5huG/k8ERF5BBBvBNzGM2xbbZN9mge2+bYbjc9H+12e2zSbI9tu9202W22+Zne2TZt0lwacyiJCgiKXIKCGlEREASRe2BmvjOf3x+DxiSiCAwzzHyejwePOOf3nQTfj8/x/r4/QkqJoiiKP9J5OgBFURRPUQlQURS/pRKgoih+SyVARVH8lkqAiqL4LYOnA7hSTEyMTE9P93QYiqL4mOrq6gtSytgPP+9VCTA9PZ2qqipPh6Eoio8RQrRc7Xk1BVYUxW+pBKgoit9SCVBRFL+lEqCiKH5LJUBFUfyWV+0CK8pEbN++naamJhISEnjggQcQQng6JGWWcmsCFEJ8GXgIkEAD8Gkp5ag7r6n4NofDwf/8/OfY7RpIJ5s3byY+Pt7TYSmzlNumwEKIZOALQL6UchmgB+531/UU/3Du3DnsNhv26PkANDc3ezgiZTZz9xqgAQgWQhiAEOCcm6+n+LimpiYAbLEZIMTlx4oyGW5LgFLKduAnwFmgA+iXUu768PuEEI8IIaqEEFXd3d3uCkfxEbW1tQhDIE5TDM6QGGpraz0dkjKLuXMKHAncA8wFkgCTEOLvP/w+KeU2KWW+lDI/NvYjt+opymUOh4OKAwewmRNB6LCHJdHY2Eh/f7+nQ1NmKXdOgW8BTkspu6WUduBlYLUbr6f4uPr6evp6e9Ei0wHQItNxOp3s27fPs4Eps5Y7E+BZYJUQIkS46hQ2AWrBRpm07du3IwyBaBGpADhDopAhkby2fTvqbBtlMty5BngQ+AtQg6sERgdsc9f1FN/W0dHBvn0lWKMXgH6seksIrLFLaD55Uq0FKpPi1l1gKeXjUsrFUsplUspPSSmt7rye4rt+/etfIxHYEpZ94Hl7zAIINLHt2WfVKFC5YepWOMXr1dfXs3v3bkbjliADTB98UWdgJDGXY01NvPXWW54JUJm1VAJUvJrFYuGHP/oRBJmxJeVc9T1azEKc5nh+/vOnOX/+/MwGqMxqKgEqXktKyZNPPsm5c+ewpK0FvfHqbxQCS/paLFYbTzzx79jt9pkNVJm1VAJUvNZzzz1HcXEx1qRcHGGJ13yvDArHkraaxsaj/PjHP1brgcqEqG4wild65ZVX+PWvf409egG2xOwJfUaLmod1dIDdu3cTFhbGo48+qjrFKNekEqDidV588UWefvpptIhURtPXwA0kMVtiNkKz8tJLL+FwOPj85z+PXq93Y7TKbKYSoOI1HA4HTz/9NC+//DJaZDoj8zaA7gaTlxBYUwpACF599VUuXLjAt7/9bYKCgtwTtDKrqTVAxSv09vbytX/5F15++WVs8ZmMzN9448nvkrEkOJpaSFlZGf/0uc/R1tY2rfEqvkElQMXjamtr+cyDD1FbW8do+hqsqYUgpv6raY/PxLLwVs60nuOhhx+muLhYbY4oHyC86RciPz9fqoPR/cfIyAjbtm3jlVdegeBwhudtxBkSPe3XEdYhQk7tRTfUxfr16/nKV75CRETEtF9H8V5CiGopZf6Hn1drgIpH7N+/n5899RTdXV3Y4pZinZP//j2+00wGhjK8eAsBnQ2UlO2nuqaWf/rsP7JlyxZ0OjUJ8mdqBKjMqJaWFn7xi19w4MABZEgkI6lFOMwJM3Z93UgvwS0V6AY7yVi8mC98/vNkZmbO2PUVzxhvBKgSoDIjenp6+O1vf8vOnTuROgMjCdnY4zNhEiOwwLMHALCmrppcMFJi6GkmuL0KbBbWrVvHww8/TGpq6uS+T/F6agqseERPTw/PP/88r732GnbNgS12MbakbKQxeNLfqbNcnFpQQqDFLGAwMo2A80cpKz9AWVkZt9xyCw888AApKSlT+35l1lAJUHGLjo4OXnzxRXbs2IFd07BHzcealIMMCvN0aO/TG7El5WCPzSCgo57id/dQXFzMpk2buP/++1mwYIGnI1TcTCVAZVodO3aMP//5z+zduxeJwB49H2titnclvg+RxmCsqYXYEpcT0NnAO3v2UVxcTN6KFfzN/feTn5+vbqnzUSoBKlNms9nYt28fL738MseamhCGAEbjl2GPX/rR/n1eTBqDsaYUYE3MJqD7GDUNTdRUf42UlFQ+/vGPcfvttxMSEuLpMJVppDZBlElra2vjzTffZMeO1xkY6IfgcEZjFmOPXQj6ALddN/jYGwCMLN7itmsA4HRguHiKwK4mdMMXCAwKYvPtt3PnnXeycOFC915bmVZqE0SZFlarlZKSEl5//XUOHz7s2lAIT8G2aBWOsKQbalzg9XR6tJiFaDEL0Q11Y+9q5LUdr/Paa6+xYMFC7rxzK5s2bcJsNns6UmWS1AhQuS6n00l9fT27du1iz569jIxYICgMa/RC7DELZnyaO2MjwKvRrBh7mgnseQ8x3IPBYGTNmtXcfvvtFBQUYDCoMYU3UiNA5YadOnWKd955h127d9Pd1YXQG7FFpGFPXYDDnOhbo72JMgRij1+KPW4JOksPxgsnKSmvZN++fZjDwrll081s2rSJzMxMtXEyC6gEqHxAW1sb7777LsXvvMPZlhbXFDcsCfu8Da7zeMdrS+9vhMBpisFqisGaUoB+oA37hZO88tp2XnnlFWLj4rhl0yZuvvlmFixYoJKhl1IJUKGlpYWSkhL27N3LqeZmAJzmeGxpRWiR6VMqWvYLOh2OiFQcEamMOmwYes/ScfEUz//pzzz//PMkJCZy08aNbNiwgYyMDJUMvYhaA/RDUkpOnjxJWVkZe/budY30AKc5DltEuivpBYZ6OMrxeXQN8AYI+yiGvhYMF89gGDwHUhIbF8dNGzeydu1aMjMzVbfqGTLpNUAhRD6wDkgCRoAjQLGUcor3IykzyeFwcOTIEUpLSykpKaWry3V8pMOcgD11FVpk2qyq2ZsNpDEIe2wG9tgM0EZdI8PeM7zw4l944YUXCAuPYN3aNaxbt468vDwCAtxXOqRc3bgJUAjxD8AXgNNANXAcCALWAl8XQhwB/k1KeXYG4lQmwWKxUFVVxf79+9lfXs7Q4KCrtMOchJa+Fi0iRU1vZ4ohCC12EVrsInDYMPS1Ye9t4Y23drFz504Cg4IoLChg7dq1FBYWEh4e7umI/cK1RoAmYI2UcuRqLwohcoCFwLgJUAgRAfwSWAZI4DNSyorJBqtcX1dXFxUVFZSXl1NdXYOm2RGGQGxhc9Dmr0ALT1EbGZ6mD0CLnocWPY9RpwP9wDlsfWcpPVBFSUkJQgiWZWWxZvVqVq9eTUpKilo3dBO3rgEKIX4HlEopfymECABCpJR9471frQHeOKfTyfHjxykvL2d/efnlTQyCwrCFp6BFpOIIjZ9U2ylvNVvWAG+YlOiGL2DoO0tAfytirOtNYmISa9euoaioiOXLl6taw0mYdD9AIcRc4PNAOleMGKWUd1/nc2HAYWCenGCWVQlwYoaHhzl06BAHDhygvOIAA/19IASO0Hjs4Sk4IlJwBoX7bJ2ezybADxHWIQz9rRj6zmIY7ASng+DgEAoKVlJUVERhYSGRkZGeDnNWmEoh9KvAr4AdgPMGrjkP6AZ+I4TIxrWO+EUp5fCHAnsEeARQDSmvoa2tzTW1raig/vBhHA7H2NQ2GW1eNlr4HDAEejpMZRrJwFDscUuwxy0Bhx3DwDlsfa2UVBxi3759CCHIyMhg9erVFBUVqXrDSZjICPCglLLwhr/YtXt8ANc64kEhxFPAgJTy38b7jBoBvs9ut1NfX8+BAwco219Ox7l2AGRwBLaxUZ4jNG5aTk+bbfxlBDguKdFZejD0tWLsb0M33A1AVHQ0a1avZtWqVeTl5REcrDa4LpnKCPApIcTjwC7AeulJKWXNdT7XBrRJKQ+OPf4L8K8TjNcv9fb2cvDgQSoqKjhYWcnoyAhCp8duTkBLXeXatQ1UN977vbG7UGymGGzJuQi7BX1fG+f723j9zbfZsWMHBoORvLxcVo8lxISEmTt3ZTaZSALMAj4F3Mz7U2A59nhcUspOIUSrECJDSnkc2AQ0TiVYXyOl5PTp05SXl1O2fz/Hjx1DSokICMEanoIjOQUtLEnt2irXJI0hl0tsRp0O9IOdGPrbqKw/RmVlJQDp6XMvb6QsWbJEnYY3ZiJT4GPAciml7Ya/3FUq80sgADgFfFpK2Tve+/1hCqxpGocPH6a8vJzSsv10ne8EwGmKwR6eghaR4jobV63ljMvvp8ATJSVidABD/1mMfa3oh86DlIRHRFwuscnPzycoKMjTkbrdVKbAh4EIoOtGLyqlrAM+clF/Y7FYOHToEGVlZewvL8cyPIzQGbCHJaKlrUaLSEUGqE7DyjQTAhkcjj04C3tCFmhWDP1t2PvO8uauYt544w2MAQEUrCxg3bq1FBUV+V0B9kQSYDxwTAhxiA+uAV6zDMbfDQ4Osn//fvbtK+HQoUOugmRjkKs2LzFVTW2VmWcIRIuejxY9//2pct9Zyqvq2L+/DJ1OR1ZWFhs3bmT9+vVER0d7OmK3m8gUeMPVnpdS7pvuYGb7FHhgYICysjL27t1LdXU1DocDAkOxRaSiRaThMMf75a7tdFNT4Gl2aVe5t4WAvhbESB9CCDKXLeOmsWQYGxvr6SinZKqF0B1SytGxx8FAvJTyzHQHORsT4MjICOXl5RQXF1NZWelKekFhrsahkek4TTFqPW+aqQToXrqRXgwXz7iSoeUiQgiWL1/OLbfcwoYNGwgL894T/sYzlQRYBay+tAkydkvbfinlyukOcrYkQKfTSW1tLW+++SYlpaXYrFYINGGLnIs9ap7axHAzlQBnjhjpx3jxFIG9p2GkD71eT0FBIVu23MGqVaswGmfHMs5UNkEMV+4ASyltY0nQ73R2dvLWW2+x8403XC3iDYFYI9PRoubhMCeopKf4HBkcji05F1tSDjrLRYwXmzlQXUdFRTnmsHA2334bd9xxB/PmzfN0qJMykQTYLYS4W0q5HUAIcQ9wwb1heQ8pJTU1Nbz08stUlJcjpcQRloRt3ka0yFTQqRvTFT8gBE5TNFZTNNY5+ej727FfeI8XX3qJF198keXLs/nrv76X1atXz6pmDROJ9LPAH4QQPx973IarMNqnaZpGcXExf3z+ec62tCCMQYwmZGGPXezV3ZIVxe2EznUrZkQKVvsohgvvUX/iGPWPPUZ0TCz33/dJ7rrrrllRXzjhdlhCiNCx9w+6KxhvWAPUNI1du3bxu9//nvOdnUhTNKNxS9Gi5qrRnpdQa4BeSDox9LUScL4R/WAHYeER/O3f3M8999zjFfck3/AmiBDi74E/Simv2gFGCDEfSJRSlk1XkJ5OgLW1tfz0Zz/jbEsLTlMMo4k5OCJS1NqeFwk8ewDjhfcAcIRE4wyJwpq6ysNRKVfSD3YS2HEYfX870dExPProP7Nx40aPdqqZzCZINFArhKjG1cqqG1dL/AXABlzrgD7R3GB4eJif/vRnFBfvhiAzIwtuRotIU4nPC+ksFxEOOwCGwU40D8ejfJTDnIDFnIB+sBPZepAnnniC7dt38K1vfZOYmBhPh/cB41blSimfAvKA54FYXM0M8oB24FNSynullO/NSJRudObMGR75x3+k+J1irEk5DGZ+DC0yXSU/RZkihzmBoSV3MZpWRF19A5958CFqa2s9HdYHXHNRS0rpAHaP/fics2fP8s///CgWuxPLos04whI9HZKi+Bahwx63BIc5Adm8h69+9av85Cc/ITc319ORAdcYAfo6i8XCv3z961jsDgYztqjkpyhu5AyOZHDxVrQAM9/69rfp6OjwdEiAHyfAXbt20dnRwfDcDcig2Xdrj6LMOoZAhhfcgmVkhJdeesnT0QB+nABLSkqQwZE4zGrkpygzRQaFYQ9LYc/eae+lMinXLWwTQgQC9/LRU+H+3X1huZ9er0fq9GqzQ1Fmmk6PXu8dY6+JRPEacA+gAcNX/MxqGRkZ6IYvoLP0eDoURfEbwj5CwEAbSxYv9nQowMRuhZsjpdzs9khm2H333ccrr7yKPF3C8KLbkUbVkVlR3MphJ/jUPoRT4zOf+YynowEmNgIsF0JkuT2SGWY2m/ne975LkGOE0ONvorNc9HRIiuKzhHUI03u7MAx28K//+q+kpaV5OiTgGglQCNEghKgH1gI1QojjQoj6K56f9XJzc/nJT/6TsACBqWk7AefqwOnwdFiK4jukxNh9HHPjqwTb+nnssce47bbbPB3VZdeaAt85Y1F4UFZWFr//3W956qn/Zs+edwnseY+RhGy06AWgjg5UlMmREkPfWYI66hDDPSzPzubrX/86SUlJno7sA8ZNgFLKFgAhxHNSyg+0vxJCPIcPtcSKiIjg8ccf4447NvPsL3/JeyfKoLOe0bil2GMWqsOLFGWinA4MvWcIPH8U3fAFEpOS+cyXvsWmTZu88iziiWyCZF75QAihB1a4JxzPKigoYOXKlVRUVPDb3/2OE8cPENxejTV6PvbYJThDIj0douKwERQUxJ133snrr7/OkOOGj6tW3EBYhzB2HyOo5z2kbYSk5GQ+9ejXufXWW726Qeq4kQkhvgF8EwgWQgxcehqwAdtmIDaPEEKweuzQ6KamJl555RXefXcPWtcxpCkaa9R8tKh56hxfDxGajTvvvpNHH30UKSUv7Hjb0yH5L82KsfcMxp5m9IOdCCEoKiri4x//OHl5eV454vuwiRyK9B9Sym/MRDCe7gc4nr6+PoqLi3l71y7eO3EChMBhTsQeNdd1qLnR8w0f/UXw0VcJdVrYunUrO3fuZEgXwkjmX3k6LP/hsGHoa8XQe4aA/jak00FScjKbb7+d2267jYSEBE9HeFVTORUu7ypP9wMtUsppbcfmrQnwSi0tLezevZvdxcWc7+x0JcPQeOyRaWgRaapdvpsFH3sDw2Dn5ceaOUF1hnYzYR9B39eKsbcFw+A5cDqIiIjk5ptv4rbbbiMjI8OjzU4nYioJ8ACuPoD1uKbAWcBhXA1TPyul3HWdz+uBKqBdSnnNneXZkAAvkVLS3NxMaWkpe/fto+XMGdfzIVHYwua4zkwIjVUHoU8zlQBngJToRi5i6GvF2N+GbqgLgLi4eDZu3MC6devIzMycFVPcS6ZyLOYZ4EEp5dGxL1oKfA34LvAycM0ECHwRaAJ8quWKEIIFCxawYMECPv3pT9Pa2kp5eTkVFRXUNzTg7KxHGIOwmZPQwpNxhCWrdUPFe2lWDAPn0Pe3EzDYDlbX3a4ZGYtZvXorRUVFLFy40OtHejdqIglw8aXkByClbBRC5EopT13vP4YQYg6wFfg+8JUpRerlUlJSuO+++7jvvvsYGhqiqqqKiooKDhw8SP/pU4BrdGgPS3YlxNA4dciS4jnSiW74Aob+dowD7eiGu0FKgkNCWFmYT1FREYWFhURFRXk6UreayN/A40KIXwB/Gnt8H3BirEuM/Tqf/RnwL4B5vDcIIR4BHgFITU2dQDjeLzQ0lI0bN7Jx40acTifNzc1UVlZSWVnJkSNHcHQ2IPQG7KZ4tLAkHGFJOEOiVGcaxX2kRIwOYBhoRz9wjoCh80jNihCCRYsyKCzczMqVK1myZIlXl61Mt4msAQYDn8N1S5wAyoD/BUaBECnl0DifuxPYIqX8nBBiI/BVX1oDnCyLxUJdXR3V1dUcqqribEsLAMIYjC00AUdYIlpYEjLQrBLiVag1wIkTNgv6gXMYBjsIGOxAWl1/VWPj4iksWMmKFSvIy8sjPDzcw5G636TXAKWUI8B/jf182FWT35g1wN1CiC24TpMLE0L8n5Ty7ycYs08KCQm5XGcI0NPTQ3V19eWEeLHltOuNQaHYQxPRzIk4wpLU+qFyfZoVw0AH+rGEx0gfAKFmMytWrSA/P58VK1Z43e1onjSREeAa4DtAGh9siDpvwhdRI8AJkVLS2tpKbW2tKynW1DI8NHYOfUjk+yNEcyIYAj0brIeoEeAVHHb0Q+cxDHRgHOxADF8AIDAwiOzs5ZdHePPnz59VO7buMJVd4F8BX8Z1NrBqleJGQghSU1NJTU3lnnvuwel0cvLkSWpqaqiuqeHw4cPYupoAcJpisI+NDh2h8aD3n3Ubv+V0oh/uHpvWnkM/3O16Tq9naWYm+SvuIi8vz+/W8aZiIv+V+qWUb07lIlLKvcDeqXyHP9LpdCxatIhFixZx//33Y7fbaWpqciXE6moaGxtxdDaATo8jNA7NnIQWloTTFK3qD32BlOhG+lwJb+AcxqFOpMN+uQQrP/9m8vLyyMrKIigoyNPRzkoTmQL/ENDjqvmzXnpeSlkz3cH4+xT4RlksFhoaGi6vH54+5Sq3EcZAbKGJOMKT0cKSferuFF+fAgv7iCvh9be7Ni5srnq8xKRkClbmk5eXR25uLmFhPlVW63ZTmQIXjv3zyg9L4ObpCEyZvJCQEAoLCyksdP0v6u3tpaamhqqqKg5WVnLxzH7XG4PDsZmT0SLm4DAnqPpDbyKd6Ie60Pe3ETDQjhh2nVFjCjVTsMbVnWjFihXEx8d7OFDfNJFd4JtmIhBl6iIjI9m0aRObNm1CSklLSwuHDh2i8tAh6mprsXc1uuoPQxPRIuaghaf41OhwthD2EfT9bRj62ggYPIfUrOh0OjKXLaOw4GPk5+ezcOFC9Hq9p0P1eRM5FjMe+AGQJKW8Y+xWuCIp5a/cHp0yaUII0tPTSU9P5xOf+ARWq5W6ujoOHDhAeUUF51sqgAqkKRpbeCpaZCrOYFWM7S5idABDXwvGvrPoh7pASiIiIim69WYKCwtZsWIFZvO49wsobjKRNcA3gd8A35JSZgshDECtlHLaD0pSa4Az41K5TXl5OaWlZTQ2HkVKCUFmbBFp2KPm4QyJ9spkOJvWAMVoP8aeUwT0nUFYegGYO28e69etY82aNSxYsMDvy1NmylTWAGOklC+MNUhFSqkJIVQ5zCx2ZbnN/fffz8WLF6moqKCkpISqqioCOo9AcDjWiHS06Pk4gyM8HfKsIaxDGC+eIqD3NGK4ByEEy7Ky2LD+71izZg2JiYmeDlG5wkQS4LAQIhrXxgdCiFW4+gEqPiIqKoqtW7eydetWBgYGKC0t5Z133qW2tgbZcRhnaBzW2Ay0yLmq3vBqnE4M/Wcxdp/A0N8GQMbixdyy6X42btxIbGyshwNUxjOR3+avANuB+UKI/UAs8NdujUrxmLCwsMvJsKenh927d7Pj9ddpP12KaD2INXohtvhMtXkCoFkJON9I0IXjSJuFyKho7vzUp7jjjjvU7WazxHXXAAHG1v0ycDVDOC6lvF4XmElRa4DeSUpJQ0MD27dv59139+BEYo+ajzVxOTJoZm+k94Y1QGG3ENBxhMALx5EOO4WrVnHP3XdTUFCg7sDwUje8BiiE+Pg4Ly0SQiClfHnaolO8mhCC5cuXs3z5ch566CFeeOEFduzYgfFiM9b4ZdgSc/xjaiydGM83EdxRCw47mzZt4m//9m+ZN2/Ct8UrXuZav7V3XeM1ievOEMXPJCQk8IUvfIG/+7u/Y9u2bbz99tsE9p5meN5NOE0xng7PbYR1EFPzu4jhHvJXFvDFL36BOXPmeDosZYqudTD6p2cyEGV2iY6O5hvf+AZbtmzh+9//Ad3H32R43kYcESmeDm3a6Sw9hL63m2Cj4OtPPMH69et9rjW8v1JFSMqUZGdn84tf/C9z01MJad6DsF6rReQs5NAwnSwmKszE/z79NBs2bFDJz4eoBKhMWXR0ND/4/vcx6AQB5+o8Hc60CuhqBOswjz/2b6Snp3s6HGWaqQSoTIuEhASWLctEP9rn6VCmlW60n8ioaJYvX+7pUBQ3mMwuMIDaBVY+YHBwkNNnzuA0+FabJmkIZKC7j/b2dpKTkz0djjLN1C6wMmVWq5Uf//jH9Pf3Y1u8ztPhTCtbfCaBF97ju9/9Hj/84X8QERHh6ZCUaaR2gZUpaW1t5bHHH+f0qVOMpqzEGepbt33JABOWtDUcO7GPBx96mCe+8zjLli3zdFjKNJlQ9aoQYiuQiet0NwCklP/urqAU79fV1cVzzz3Hzp1vIPVGLAtv9ckSGAAtKp3hwFA4tYdHH32U9es38OCDnyEtLc3ToSlTNJF+gM8AIcBNwC9x3Qdc6ea4FC8kpeT48ePs3LmTN998C83pwBazCFtijs8f2+k0xTC49B4COo9Qur+c0tISNmzYwF133UVubq5qazVLTWQEuFpKuVwIUS+lfEII8V+o9T+/0t/fz969e3lt+3ZONTcj9AaskfOwJWW7DnD3F/oAbMl52OOWYuxsYN/+Cvbu3UtCYiJ33Xknt956K3FxcZ6OUrkBE0mAI2P/tAghkoAeYK77QlK8QU9PD2VlZezdt4/DdXU4nU6kKRprWhH2qHl+ey4xgDQGYUtZiS05F0NvC+3dx3n22Wd59tlnyVi8mJs2bmT9+vWqI8wsMJEE+LoQIgL4T6AG1w7wL90ZlDLzLp1BXFlZSUXFgfe7RAeHY41fhhY513XcpvI+nQEtej5a9HzE6ADGi6dpam3h+DPP8MwzzzBv/nxWFxVRUFDA0qVLVacYLzSRlviBUkrrpT/j2ggZvfTcdFLtsGbWpVPkKisrOXDwIP19fQBXnBOS7uoG7UW3fnlDO6zrEdZBDL1nPnD+R3BICCvz8y+f/5GQkODpMP3KVFriVwB5AGNJzyqEqLn0nDJ7DA4OUldXR11dHVXV1bScOQOAMAZhMyehzc3CEZ6MNPr2hoa7yUAz9oQs7AlZoFkxDHRg62+j5GA1JSUlAMQnJJC/YgV5eXnk5OQQHa1G155wrTtBEoBkIFgIkYurGSpAGK5dYcXL9ff309DQQH19PbW1tZw8eRIpJUJvQDPFoc1ZgWZOdLWxEmoX0y0MgWhR6WhR6VilRDfSh37wHG0DHXS9XczOnTsBmJOSSl5uDtnZ2WRnZxMT47utxbzJtUaAtwP/AMwBnrzi+QHgm9f7YiFECvB7IAFwAtuklE9NOlLlunp6emhoaBgb5R3mzJnTrhd0ehymWLTEHBxhiThMsaBTZ87OOCFwhkTiDInEHp/JqHSis1zEMHCOMwMdtO98k+3btwOQmJhEbm4Oy5cvJzs7m4SEBNWFxg0msgZ4r5TypRv+YiESgUQpZY0QwgxUA38lpWwc7zNqDXDipJR0dnZSX1/vGuHV1XGuvR3g/RGeOQGHOQGHKQZ0vrEAPxvWACdtLCHqBzvQD54nYPg80u5aao+OiSU3J/tyQkxNTVUJ8QZMZQ1wvxDiV9zgwehSyg6gY+zPg0KIJlxT6nEToDI+KSXt7e3U1dVx+PBhamrr6LnQDYAwBGI3xaHNWYnDHI8zJAZUYe7sI3Q4TTE4TTHYE7IYlRLdyEX0g+fpHDzPOyXlFBcXA2AOCycnezm5ubnk5OSQnp6uirEnYSIJ8DdjP98ae3wC+DNwzQR4JSFEOpALHLzKa48AjwCkpqZO9Cv9Qnt7O7W1tdTV1VFTW8vFnh4AREAwNlM8jtT5OMwJOIMjvWqnVpkmQuAMicYZEo09fimjUiKsA+gHz2Mf7KT0UB2lpaUAhJrN5ObkkJOTQ25uLnPnzlUjxAlw+8HoQohQ4CXgS1LKgQ+/LqXcBmwD1xR4ot/riwYHB6mpqaGqqoqDlYfoOu+a6omAEGyh8TjSMlwJLyhcJTx/JAQyKBwtKBwtdhGjuEpu9IOdYwnx8OWEGBERSUHBSvLz81mxYoXaZR6HWw9GF0IYcSW/P6j+gR8lpaSlpYWysjJKy8o4cfy4a5fWEIAtNAFH6iocYUkq4SnjkoFmtEAzWszCsYQ4hH7gHPaBdnbvKWXXrl0ApKfPZe3aNaxZs4aMjAw1XR7jtoPRhWv8/SugSUr55PXe7y+klDQ2NlJSUkJJSSkdHecAcIbGYk/MxhGWPLZLq35BlRsnA0PRYhe5RohSju0yt9N8sZ0zf/gD//d//0dkVBTr1q5l/fr15Obmotf7b0WA2w5GF0KsBUqBBlxlMADflFK+Md5nfHkXuK+vj127drFjx+u0tp4FnQ7NnIgWkYYWkerz3VSmi0/vArubZsXQ14qh7ywBA+1Ih53omFju3LqFO+64w6fvThlvF3giZTBBwOeAtbimwaXAM1LK0ekO0hcTYFdXF7/61a8oLi7G4XDgDI3DFrMIe2Q6GAI8Hd6soxLgNHFqGPpaMV44gaG/HYRgVWEhjzzyiE8e9D6VMpjfA4PA/4w9/hvgOeAT0xee77FarfzhD3/g+T/9CU1zYI3JwB6X4dqxVRRP0xnQouaiRc1FWIcwXjjBwapaDh58kLvuuosHH3yQ8PBwT0fpdhNJgBlSyuwrHu8RQhx2V0C+QNM0HnvscQ4ePIA9ai7WOfn+1TdPmVVkYOjlPocB5+rYvmMHDUeO8PP/+R9MJpOnw3Oriay0147t/AIghCgE9rsvpNnvN7/5DQcPHmA0bTWj829SyU+ZFaQxCGvaKiwLb+X06TP88Ic/9HRIbjeREWAh8IAQ4uzY41SgSQjRAEgppTow9UNOnDiB0xSNPW6xp0PxOc6QKKTFVRDuCInGGRLl4Yh8jyN8DvbINI6fOOHpUNxuIglws9uj8DHR0dHoR2vRD3biMPvuzponWFNXobNcBFCbH24iRvsxDp0nZoHvN36/bgKUUrbMRCC+5OGHH+bI0aO0v7ebkeQV2GMzVPcVxftJiaH3NCGtlYQGGfna177q6YjcTlXbukF0dDRP/exnLMtcQtDZA5iPvozhwnvgnPAdhIoyc6QTfd9ZQpu2E9y8l9SkOJ762U+ZO1eNAJVJio6O5r+feorKykr+37ZtnGouRbRVYo2Yiz1mAU5TrLq9TfEo3UgvhgsnCew9BdZh4uLieeiL32TTpk1+c3eISoBuJISgsLCQlStXUl1dzVtvvUVJaSn27mMQHI4tPAUtPAVHaLy69U1xPynRDV/A0N9KQH8bYvgCOp2OgoICNm/ezJo1azAajZ6OckapBDgDdDodK1euZOXKlQwPD7Nv3z6Ki9/h8OE6HJ1HEIZA15kcEXNwhCUhA3y79kqZQdoohoEODP1trtvfbBaEECxesoSbb7qPW265hchI/y3OVwlwhplMJrZs2cKWLVuwWCxUV1dTUVFBeXkFfafHWtgHh7u6wZgTcYQlqEOKlInTrBgGO9EPdmIc6kQMu0qGgkNCWLWmkKKxYzojIiI8G6eXUAnQg0JCQli3bh3r1q27fC5vbW0ttbWurs8j3cddbwyOwBYahyM0HkdovKuw2o/XD1Xt3/uEbRj90Hn0g10Yh7sQwxcAMBqNZC5bRl7uPeTk5KhziccxoW4wM8UXmyFMlqZplxNiXV0dDUeOYBkeBsYapJpixxJiHM6QaFVm4w+kE91IL/qhLvSD5zFaumF0EICAgECWLl1CzlhX6CVLlhAYGOjhgL3HpLvBzCSVAMfndDo5c+YMR44c4ciRIxyur+d851hXFJ0eZ0g0mikWR2gsDlOcax3Rj0eJvkDYR9ANdaMf7sIw1I3BcgHpcHWii4iMInt5FsuWLSMrK4sFCxaoEd41qATog3p6emhsbOTo0aMcOXqU48eOY7fbABCBJmwhMThMcThDY3GExIBe/QXxWk6H60S44W70Q10YLRdg1HWChE6vZ/78+WQtW8bSpUvJzMxUx2TeIJUA/YCmaTQ3N3P06FEaGxs5cuQonZ0drheFQIZEYQ8ZGyWGxiEDw9Qo0ROkdK3dDXehH+rGMNyNztJzuVA+MiqKZZmZZGZmsnTpUjIyMtR0dopUAvRTfX19NDY2jv000djUyOjICADCGPSBhOgwxYLev+rAZoTTgc7S41q7G+oiwNKNtLrWc41GIxkZGSxduvTyT1xcnIcD9j0qASoAOBwOWlpaLifF+voG2tpaXS9eGiWaxnaczfGqJnEShH0U3VAXhqHzGIbOf2B0FxsXz/KsZZdHd2rtbmaoBKiMa2BggKamJo4ePUrDkSM0Hm3Eah078SDIfEVCVEdyXo2wDqEf7EA/dJ6A4W6w9AKgNxhYnLGYrCsSnjqe0jOm0hJf8XFhYWEUFhZSWFgIvL+W2NDQQENDA3WHD9Pf0gxccUaxOREtLNEv1xGFbRj9QAf6wQ4Chs5f3qwIMZlYvjyLrCzXj1q7835qBKhcl5SS9vZ26urqqKuro6q6hr5eV08+EWhy3cYXlowWngSGIA9H6wYOO/rBDgz97QQMnoMR17HYJlMoubk55ObmkpOTw9y5c9V5u15KTYGVaSOlpLW1lbq6Ompqajh0qIrh4SFg7HzjsGS08BScppjZOTqUEt1on+sIyYF29EPnwekkICCQvLxcVqxYQU5ODvPnz1cJb5ZQCVBxG4fDwbFjx6isrORgZSXHjx1DSgmBoa6ON5Fprs7YwouTxaVOKb0tBPa3XB7lpafPZdWqQgoKCli2bBkBAeoo09lIJUBlxvT393PgwAFKSkqprDyI3W5HGIOwRqR7XS9EnaUXQ89JAntPg3UInV5PTnYO69evY/Xq1aokxUeoBKh4xMjICIcOHWLfvn2UlpZhs1khOBxr1HzsMQs9U2ajWTFeOEngxWZXTzy9nsKCAm666SaKioowm9Upfr5GJUDF4ywWCyUlJbz19tvU1dUBYI9MxxafOSOjQt1IL8bzjQRebEY6NBYuWsQdmzdz8803q/ZQPs4jCVAIsRl4CtADv5RSXvOgUZUA/UdHRwevvvoq23fsYMRiwRkax2hSLo7w5Gm/lm64h8D2agz9bRgMRm677Vbuvfde5s+fP+3XUrzTjCdAIYQeOAHcCrQBh4C/kVI2jvcZlQD9j8Vi4e233+YPf/wjF7q7cYQlMpqcjzM0dsrfLUYHCGyrwth7BpMplPvu+yR33323Gu35IU8UQhcAJ6WUp8YC+BNwDzBuAlT8T0hICB/72MfYunUrO3bs4He/+z0DTTuwxWdiTc6b3L3J0klA5xECz9USaDRy3wMP8MlPfpLQ0NDp/xdQZjV3JsBkoPWKx21A4YffJIR4BHgEIDU11Y3hKN4sICCAe++9l82bN7Nt2zZee+01AvpbGZ53E07TxG8fE9YhQpr3oBvuZu3atXzpS18iJibGjZErs5k7C7OutqL9kfm2lHKblDJfSpkfGzv1aY8yu5lMJr785S/z1FNPEW0KIPT4TgwXT03os/rBTszHdmByDvOd73yH733veyr5KdfkzgTYBqRc8XgOcM6N11N8SHZ2Ns8+u42lixcT3LwXQ0/zNd+vH+zE9N4uEmOieOaZX7Bx48aZCVSZ1dyZAA8BC4UQc4UQAcD9wHY3Xk/xMVFRUTz55H+RnZND8OkS9P1tV32fztKL6b3dJCcl8vTTPyctLW2GI1VmK7etAUopNSHEo8DbuMpgfi2lPOqu6ym+KTAwkP/4wQ/43D8/SsvpEixzVvLh1ZWgznrCzCZ++uSTfn3GrXLjVCG0Miu0tLTw2c/+EyMjlo+8ptfr+dGPfkR+/keqHBQFUP0AlVkuLS2Nv/zlRfr6+j7ymslkUrV9yqSoBKjMGiaTCZNJtehXpo8X9ydSFEVxL5UAFUXxWyoBKorit1QCVBTFb3lVGYwQohto8XQcyqwQA1zwdBDKrJEmpfzIvbZelQAVZaKEEFVXq+tSlBuhpsCKovgtlQAVRfFbKgEqs9U2TwegzH5qDVBRFL+lRoCKovgtlQAVRfFbKgEqiuK3VAJUFMVvqQSoKIrf+v9rbTiKdsALCwAAAABJRU5ErkJggg==\n",
      "text/plain": [
       "<Figure size 360x216 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "fig, ax = plt.subplots(figsize =(5, 3)) \n",
    "sns.violinplot( ax = ax, y = df[\"petal length (cm)\"] ) \n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 31,
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<AxesSubplot:ylabel='petal width (cm)'>"
      ]
     },
     "execution_count": 31,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 360x216 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "fig, ax = plt.subplots(figsize =(5, 3)) \n",
    "sns.violinplot( ax = ax, y = df[\"petal width (cm)\"] ) \n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### The difference between the boxplot and the violin plot"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "- A violin plot is a mrthod of ploting numeric data. it is similar to box plot, with the addition of a rotated kernel density plot on each side.\n",
    "- violin plot are more informative than a plain box plot.\n",
    "- Box plot only shows summary statistics such as mean, median and interquartile ranges. \n",
    "    But violin plot shows the full distribution of the data.\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### [Problem 7] Confirming the relationship between features"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 32,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "df.plot(kind='scatter', x='sepal length (cm)', y='sepal width (cm)') \n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 33,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "df.plot(kind='scatter', x='petal length (cm)', y='petal width (cm)', color='red') \n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 34,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "df.plot(kind='scatter', x='sepal width (cm)', y='petal length (cm)', color='green') \n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 35,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "df.plot(kind='scatter', x='sepal length (cm)', y='petal width (cm)', color='blue') \n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 36,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "df.plot(kind='scatter', x='sepal width (cm)', y='petal width (cm)', color='purple') \n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 37,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "df.plot(kind='scatter', x='sepal width (cm)', y='petal length (cm)', color='skyblue') \n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 38,
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAAYMAAAEGCAYAAACHGfl5AAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjMuMiwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy8vihELAAAACXBIWXMAAAsTAAALEwEAmpwYAAAhWUlEQVR4nO3df7QkdXnn8fcnMCxXcMDs3CMqONcYw87gijBzBnCQsNEZ+WEkiUpAjTvKHlaXO0eWuDma7OqGJObEuNkcwTghDskozLiKqARBxiSoMCjMHeTnvaKoww8xzPVHBtSJMPHZP6ou9O3pvlU9XVVd1f15ndOnp6uqq57+3jv13K76Pt+vIgIzMxttvzDoAMzMbPCcDMzMzMnAzMycDMzMDCcDMzMDDhx0AL1asmRJTExMDDoMM7NG2bFjx/cjYrzb+sYlg4mJCaampgYdhplZo0h6YKH1vkxkZmZOBmZm5mRgZmY4GZiZGU4GZmZGiclA0lGSbpQ0I+leSe/osM2pknZLuiN9vKeseMwaYXYWtm9Pnkc5hryaFGvNlfnNYC/wuxGxDDgRuEDS8g7b3RQRL00fF5cYj1m9bdkCS5fCmjXJ85YtoxlDXk2KtQFU1RDWkj4LXBoRX2hZdirwzoh4dd79rFy5MlxnYENndjY5oe3Z8/SysTF44AEY71onNHwx5NWkWGtC0o6IWNltfSX3DCRNAMcBt3ZYfZKkOyVdL+mYLu8/X9KUpKlZfx20YbRzJxx00PxlixYly0cphryaFGtDlJ4MJB0KfAq4MCIea1t9O7A0Io4FLgE+02kfEXFZRKyMiJXjzvo2jCYm4Ikn5i978slk+SjFkFeTYm2IUpOBpEUkieDKiLi6fX1EPBYRP07/fR2wSNKSMmMyq6Xxcdi4MbnUsXhx8rxxY7WXPOoQQ15NirUhSrtnIEnAJuCHEXFhl22OAB6NiJC0CriK5JtC16B8z8CG2uxscqljYmJwJ7Y6xJBXk2IdsKx7BmUOVLca+B3gbkl3pMt+H3g+QERsAF4HvF3SXmAPcM5CicBs6I2PD/6kVocY8mpSrDVXWjKIiJsBZWxzKXBpWTGYAc3667EOsdYhBqucK5BtuDWpL3odYq1DDDYQldUZFMX3DCy3JvVFr0OsdYjBSlOLOgOzgWhSX/Q6xFqHGGxgnAxseDWpL3odYq1DDDYwTgY2vJrUF70OsdYhBhsY3zOw4dek3jF1iLUOMVjhBllnYFYPTeqLXodY6xCDVc6XiczMzMnArDJFTMRSxWQueY6Rtc3MDGzalDxbI9rDycCsCkUUc1VREJbnGFnbrF8Py5fDunXJ8/r1xcfZJA1pD99ANitbEcVcVRSE5TlG1jYzM8kJr930NCxbVkycTVKj9nDRmdmgFVHMVUVBWJ5jZG1z222d991t+bBrUHs4GZiVrYhirioKwvIcI2ubVas677vb8mHXoPZwMjArWxHFXFUUhOU5RtY2y5bB5OT8/U5OjuYlImhUe/iegVlViijmqqIgLM8xsraZmUkuhaxaVcsTX+Vq0B5Z9wycDMzMRoBvIJsNk6z+/VXUIVjvGvBzcTIwa4qs/v2emKaeGvJz8WUisybI6t/viWnqqUY/F18mMhsGWf37PTFNPTXo5+JkYNYEWf37PTFNPTXo5+JkYNYEWf37PTFNPTXo5+J7BmZNktW/3xPT1FMNfi6e3MZsmGRNPOOJaeqpAT8XXyYya5KmzGfQlLkbitCUODM4GZg1RVPmM2jK3A1FaEqcOfiegVkTNGU+g6bM3VCEpsSZcp2B2TBoynwGTZm7oQhNiTMnJwOzJmjKfAZNmbuhCE2JMycnA7MmaMp8Bk2Zu6EITYkzJ98zMGuSpsxn0JS5G4rQkDg9n4GZmQ3uBrKkoyTdKGlG0r2S3tFhG0n6oKT7Jd0l6fiy4rGGKqLP+yiZmYFNm5Jnq48G1F2Uec9gL/C7EbEMOBG4QNLytm1OB16UPs4HPlxiPNY0RfR5HyXr18Py5bBuXfK8fv2gIzJoTN1FZZeJJH0WuDQivtCy7K+BL0bElvT1fcCpEfG9bvvxZaIRUUSf91EyM5MkgHbT056DeJBqVHdRizoDSRPAccCtbaueBzzU8vrhdFn7+8+XNCVpataXAkZDEX3eR8ltt/W23KrRoLqL0pOBpEOBTwEXRsRj7as7vGWfryoRcVlErIyIleOj9hffqCqiz/soWbWqt+VWjQbVXZSaDCQtIkkEV0bE1R02eRg4quX1kcAjZcZkDVFEn/dRsmwZTE7OXzY56UtEg9aguovS7hlIErAJ+GFEXNhlmzOBSeAM4ATggxGx4J8yvmcwYoro8z5KZmaSS0OrVjkR1EkN6i4GVmcg6WTgJuBu4Ofp4t8Hng8QERvShHEpcBrwU+AtEbHgmd7JwMysdwOb3CYibqbzPYHWbQK4oKwYzMwsH49NZMOvSYVYVUwaY/O5TQEnAxt2TSrEqmLSGJvPbfoUj01kw6tJhVhVTBpj841Ym9ai6MxsIJpUiFXFpDE2n9t0HicDG15NKsSqYtIYm89tOo+TgQ2vJhViVTFpjM3nNp3H9wxs+DWpEKuKSWNsvhFp04HVGZjVxrJl9U8Cc8bHFz4hZa233rlNAV8mMktk1SIUMclOnn30G0cDJlEplOsyihMRjXqsWLEizAo1ORkBTz8mJ+ev37w5Ymws4rDDkufNm/fdR9Y2efbRbxx5jpGliH1UpYr2GCLAVCxwbh34yb3Xh5OBFWp6ev4JeO4xPZ2s37UrOZG0rhsbS5bPydomzz76jSPPMbIUsY+qVNEeQyYrGfgykY22rFqEIibZybOPfuNo0CQqhXBdRuGcDGy0ZdUiFDHJTp599BtHgyZRKYTrMgqXKxlIepakYyT9kiQnEBseWbUIRUyyk2cf/cbRoElUCuG6jMJ1rTOQdBjJ8NLnAgcBs8DBwLOBrwJ/FRE3VhTnU1xnYKXIqkUoYpKdPPvoN44aTKJSKddl5Lbfk9tI+gLwUeDvI+Jf2tatAH4HuDsiNhYXbjYnAzOz3u130VlErFlg3Q5gR5+x2f7yXztWJv9+jaS89wxeIuk1kn5r7lF2YNaFx1+3Mvn3a2Rljk0k6XLgJcC9PD2XcUTEW0uOraORvkw0YuOvW8X8+zXUihib6MSI6DBDiFVuru9063/Wub7T/s9q/fLv10jLc5noK5KcDOrAfaetTP79Gml5ksEmkoRwn6S7JN0t6a6yA7MO3HfayuTfr5GW557B/cBFwN08fc+AiHig3NA6G+l7BnPc28PK5N+voVTEPYMHI+KaAmOyfnn8dSuTf79GUp5k8HVJm4G/B342tzAiri4tKrO6qWq2NP9V3hu3V2Hy3DMYI0kCa4FfTx+vLjMos1pZvx6WL4d165Ln9evLOY77+PfG7VUoz4FstpCZmSQBtJueLvYbgvv498bt1bOsewaZ3wwkbZJ0eMvrZ6WFaGbDL2uegaJ4/P3euL0Kl+cy0UtaB6qLiB8Bx5UWkVmdZM0zUBT38e+N26tweZLBL0h61twLSb9IvhvPZs2XNc9AUdzHvzdur8LlqTN4M/Bu4CoggLOBP4mIj5Uf3r58z8AGwr2J6sntldt+z2fQtpPlwK8BAv4xIqaLC7E3TgZmZr3b76IzSYdGxI8B0pP/PgmgdZsO6y4n6YK6KyJe3GH9qcBnge+ki66OiIu7fxQzMyvLQvcMPivp/0g6RdIhcwvTeZDPk3QDcNoC7/+7jPUAN0XES9OHE8GwmZ2F7duT50Huo4jjzMzApk3Jc1nHyHOcrH3Upc2r+rlZcSKi6wM4A7gS2Ak8BvwAuAX4A+CIhd6bvn8CuKfLulOBa7P20f5YsWJFWANs3hwxNhZx2GHJ8+bNg9lHEceZnIyApx+Tk8UfI89xsvZRlzav6udmPQGmYqHz9UIr+33kSAY/AO4ErgeOybNPJ4MG2LUrOQm0ntjGxpLlVe6jiONMT89fN/eYni72s2QdJ2sfdWnzqn5u1rOsZJBr2suS3A4sjYhjgUuAz3TbUNL5kqYkTc36a2f9FVEQVFVRUdZxiig6y/NZso6TtY+6tLmLwRprYMkgIh6Lp29QXwcskrSky7aXRcTKiFg57u5j9VdEQVBVRUVZxymi6CzPZ8k6TtY+6tLmLgZrrIElA0lHSFL671VpLD8YVDxWoCIKgqoqKso6ThFFZ3k+S9ZxsvZRlzZ3MVhj5a0zOAB4Ni1dUSPiwYz3bCG5L7AEeBR4L7Aofe8GSZPA24G9wB7gooi4JSsW1xk0SBEFQVUVFWUdp4iiszyfJes4WfuoS5u7GKx2+i46k7Se5ET+KE/PdBYR8ZLCouyBk4GZWe+KmOnsHcDREeFLOKPGf931Zts22LoV1q6F1asHHU13RXy78LeHoZPnnsFDwO6yA7Ga8cQhvVm7Fk4+GS6+OHl+1asGHVFnWT/XPD/3In43/PtVO10vE0m6KP3nMcDRwOeYP+3lX5QeXQe+TFQBTxzSm23bkgTQ7uab6/UNIevnmufnXsTvhn+/BqKfyW2emT4eBL4AHNSy7NAig7SacV/x3mzd2tvyQSmiVsG1CEOr6z2DiPhDAEmvj4hPtq6T9PqyA7MBcl/x3qxdm1we6rS8ToqoVXAtwtDKc8/g3TmX2bBwX/HerF6974m/jjeRi6hVcC3C0FronsHpJAPVnQ38v5ZVi4HlEVHwvH/5+J5BhdzbozfuTVRsHFao/a4zkHQsyVzHfwi8p2XV48CNkcyFXDknAzOz3u13nUFE3AncKenKiHiylOjMsipu8/y1XZe/yKv4i9t/kVtZug1nCtwN3NXtsdBQqGU+PIT1EMkav3/Nmvnr167ddx95tqlCEXMNNGW+AmskMoawXugy0dL0nxekzx9Ln98I/DQGNDOZLxMNiZkZWL583+XT08k3hDx99+vSv7+K/vvu32992u86g4h4ICIeAFZHxO9FxN3p411ATcsrrTGyxu/P03e/Lv37q+i/7/79VrI8XUsPkfTUn1+SXgYcssD2Ztmyxu/v1ke/dXmebapQRf999++3kuVJBucBH5K0U9JO4K+At5YalQ2/rPH78/Tdr0v//ir677t/v5Us13wGAJIWp9sPdNA63zMYMu5NVPw++o3ThlI/dQZviogrWgasmyc8UJ2ZWWP0M1Dd3H2BZ3Z5mPVvdha2b0+ey9rHzAxs2pQ8l6mIz2I2KAv1O02/NRyctU2VD9cZDJEq+s1n1TIUxf33rebY3zqDOZLuJ5ny8ibgy8C2GOB9A18mGhJV9JvPqmUoivvvWwP0c5kIgIj4ZeBckorkV5MMUXFHYRHaaKqi33xWLUNR3H/fhkDmHMiSjgRWAy8HjgXuBW4uOS4bdlX0m8+qZSiK++/bEMhTZ/AgcCFwfUScFBFnRsSflhuWDb0q+s1n1TIUxf33bQjkuWdwLHAycArwfOCbwJciYmP54e3L9wyGTBX95rNqGYri/vtWY/tdZ9C2k0NJEsLLgTcBERETRQXZCycDM7Pe7fd8Bi07mAL+HXALyb2CUyIZwM7MzIZEZjIATo8IV9EMm7pc0qhi+AUzy5Sna6kTwbDZsiXpF79mTfK8ZUs946hLnGYjIPdAdXXhewZ9qkuBVBWTuZjZU/ouOrMhU5cCqSomczGz3LreM5D0Wwu9MSKuLj4cK11dCqSqmMzFzHJb6JvBry/weHX5oVkp6lIgVcVkLmaWm+8ZjKq69NJxbyKzSvRdZ5Du5EzgGODguWURcXH/4dnAjI/X4+SaFUdd4jQbcpk3kCVtAH4bWA8IeD2wNMf7Lpe0S9I9XdZL0gcl3S/pLknH9xj76MoziUpVE7r0q4rJbao4hlnTLTTZQXoJ6a6250OBrTnedwpwPHBPl/VnANeTJJgTgVuz9hme3CbfJCpVTejSryomt6niGGYNQMbkNnmSwa3p81eB55IMTfHNrPel75lYIBn8NXBuy+v7gOdk7XOkk8GuXcnJqPVEPzaWLJ8zPT1//dxjenpwcXeS57P0u48qjmHWEFnJIE+dwbWSDgf+HLgd2Al8fD+/iLR6HvBQy+uH02X7kHS+pClJU7Oj/DU9T9/7qiZ06VcVk9tUcQyzIZHnBvL7I+JnwKckXUtyE/lfCzi2Oizr2LUpIi4DLoOkN1EBx26mPH3vq5rQpV9VTG5TxTHMhkSebwZfmftHRPwskvmPv7LA9nk9DBzV8vpI4JEC9ju88vS9r2pCl35VMblNFccwGxJd6wwkHUFy2eYK4A08/Zf8YmBDRPyHzJ1LE8C1EfHiDuvOBCZJbiSfAHwwIjL/fHWdAfn63lc1oUu/qpjcpopjmNXcfk9uI+k/A+uAlUDr2fcxYFNkDEchaQtwKrAEeBR4L7AIICI2SBJwKXAa8FPgLRGReZZ3MjAz691+F51FxCZgk6TXRsSnej1wRJybsT6AC3rdr5mZFS/PPYNtkjZKuh5A0nJJ55Ucl5mZVShPMvhb4AaSGgOAbwAXlhWQmZlVL08yWBIRnwB+DhARe4F/KzUqMzOrVJ5k8BNJ/560BkDSicDuUqMyM7NK5Sk6uwi4BnihpG3AOPC6UqMyM7NKZSaDiLhd0q8CR5PUGtwXEU+WHpmZmVUmMxlIOhj4b8DJJJeKbpK0ISKKGJLCzMxqIM9loo8CjwOXpK/PBT5GMq+BmZkNgTzJ4OiIOLbl9Y2S7iwrIDMzq16e3kRfS3sQASDpBGBbeSGZmVnV8nwzOAF4s6QH09fPB2Yk3U0yqsRLSovOzMwqkScZnFZ6FGZmNlB5upY+UEUgZmY2OHnuGZiZ2ZBzMjAzMycDMzNzMjAzM5wMzMwMJwMzM8PJwMzMcDIwMzOcDMzMDCcDMzPDycDMzHAyGF6zs7B9e/JsZpbByWAYbdkCS5fCmjXJ85Ytg47IzGrOyWDYzM7CeefBnj2we3fyfN55/oZgZgtyMhg2O3fCQQfNX7ZoUbLczKwLJ4NhMzEBTzwxf9mTTybLzcy6cDIYNuPjsHEjjI3B4sXJ88aNyXIzsy7yTHtpTXPuufDKVyaXhiYmnAjMLJOTwbAaH3cSMLPcSr1MJOk0SfdJul/SuzqsP1XSbkl3pI/3lBmPtXAdgpm1KC0ZSDoA+BBwOrAcOFfS8g6b3hQRL00fF5cVj7VwHYKZtSnzm8Eq4P6I+HZEPAF8HDirxONZHq5DMLMOykwGzwMeann9cLqs3UmS7pR0vaRjOu1I0vmSpiRNzfqk1R/XIZhZB2UmA3VYFm2vbweWRsSxwCXAZzrtKCIui4iVEbFy3DdF++M6BDProMxk8DBwVMvrI4FHWjeIiMci4sfpv68DFklaUmJM5joEM+ugzK6l24EXSXoB8F3gHOANrRtIOgJ4NCJC0iqS5PSDEmMycB2Cme2jtGQQEXslTQI3AAcAl0fEvZLelq7fALwOeLukvcAe4JyIaL+UZGVwHYKZtVDTzr0rV66MqampQYdRf7OzC//lPzMDt90Gq1bBsmXlxLBtG2zdCmvXwurV5RwDsj+rmSFpR0Ss7LbeYxMNo6w6gvXrYflyWLcueV6/vvgY1q6Fk0+Giy9Onl/1quKPAa6ZMCuIvxkMm9nZ5KS4Z8/Ty8bG4IEHkr+aZ2aSBNBuerq4bwjbtiUJoN3NNxf7DSHrs5rZU/zNYNRk1RHcdlvn93Vbvj+2bu1t+f5yzYRZYZwMhk1WHcGqVZ3f1235/li7trfl+8s1E2aFcTIYNll1BMuWweTk/PdMThZ7E3n16n1P/GXcRHbNhFlhfM9gWLk3kZm1yLpn4GRgZjYCfAO5SHnmAChinoAq9jEzA5s2Jc9N57kZzPrmZJBXnv7sRfR5r2IfVdQZVMV1BmaF8GWiPPL0Zy+iz3sV+6iizqAqrjMwy82XiYqQpz97EX3eq9hHFXUGVXGdgVlhnAzyyNOfvYg+71Xso4o6g6q4zsCsME4GeeTpz15En/cq9lFFnUFVXGdgVhjfM+hFnv7sRfR5r2IfVdQZVMV1BmaZXGdgZma+gWxmZtmcDFpt2wbvfW/yXNY+8hR7FbGPK6+Es85KnvfnGHmOU0RhWxXFby5KM8sWEY16rFixIkqxZk0EPP1Yu7b4fUxOzl8/OVnOPo48cv42Rx3V+2fNOs7mzRFjYxGHHZY8b97ce5x5tulXVpxmIwKYigXOrQM/uff6KCUZ3Hzz/JPS3OPmm4vbx/R05/XT08Xu44orOm9zxRX5P2vWcXbtSk6srevGxpLleePMs02/suI0GyFZycCXiaCYyViy9pGn2KuIfXziE523mVue57NmHaeIwrYqit9clGaWm5MBFDMZS9Y+8hR7FbGPs8/uvM3c8jyfNes4RRS2VVH85qI0s/wW+tpQx0dp9wzWrp1/OWF/7hlk7SPPNfIi9nHUUfO3ab9nkOez5r1nsHhxM+4ZdIvTbESQcZnIdQatipiMJWsfeYq9itjHlVcml4bOPhve+Mbej5HnOEUUtlVR/OaiNDMXnZmZmYvOntakvuZ1mZimSW1mZn0ZjWTQpAlQ6jIxTZPazMz6NvyXiZo0AUpdJqZpUpuZWS6+TNSkvuZ1mZimSW1mZoUY/mTQpL7mdZmYpkltZmaFGP5k0KQJUOoyMU2T2szMCjH89wzmNKmveV0mpmlSm5nZgrLuGRxYZTADNT7enBNaVqzLllUzO1mT2szM+lLqZSJJp0m6T9L9kt7VYb0kfTBdf5ek48uMx8zMOistGUg6APgQcDqwHDhXUnu/yNOBF6WP84EPlxWPmZl1V+Y3g1XA/RHx7Yh4Avg4cFbbNmcBH03HUfoqcLik55QYk5mZdVBmMnge8FDL64fTZb1ug6TzJU1Jmpr10AhmZoUrMxmow7L2rkt5tiEiLouIlRGxctw3NM3MCldmb6KHgaNaXh8JPLIf28yzY8eO70t6oJAI988S4PsDPH4vmhKr4yxWU+KE5sQ6DHEuXeiNZSaD7cCLJL0A+C5wDvCGtm2uASYlfRw4AdgdEd9baKcRMdCvBpKmFuqrWydNidVxFqspcUJzYh2FOEtLBhGxV9IkcANwAHB5RNwr6W3p+g3AdcAZwP3AT4G3lBWPmZl1V2rRWURcR3LCb122oeXfAVxQZgxmZpZt+McmKt5lgw6gB02J1XEWqylxQnNiHfo4Gzc2kZmZFc/fDMzMzMnAzMycDBYk6QBJX5N0bYd1p0raLemO9PGeAcW4U9LdaQz7jO1dp8EAc8RalzY9XNJVkr4uaUbSSW3ra9GmOeKsS3se3RLDHZIek3Rh2zYDb9OccdalTf+7pHsl3SNpi6SD29b33p4R4UeXB3ARsBm4tsO6UzstH0CMO4ElC6w/A7iepNr7RODWGsdalzbdBPyX9N8HAYfXsU1zxFmL9myL6QDgn4GldWzTHHEOvE1Jhuz5DjCWvv4EsK7f9vQ3gy4kHQmcCXxk0LH0yYMB9kDSYuAUYCNARDwREf/SttnA2zRnnHX0CuBbEdE+isDA27RNtzjr4kBgTNKBwDPYd+SGntvTyaC7vwR+D/j5AtucJOlOSddLOqaasPYRwFZJOySd32F9rsEAK5IVKwy+TX8JmAX+Nr1E+BFJh7RtU4c2zRMnDL49250DbOmwvA5t2qpbnDDgNo2I7wIfAB4EvkcycsPWts16bk8ngw4kvRrYFRE7FtjsdpKvkMcClwCfqSK2DlZHxPEkc0NcIOmUtvW5BgOsSFasdWjTA4HjgQ9HxHHAT4D2iZnq0KZ54qxDez5F0kHAa4BPdlrdYdlAfk8z4hx4m0p6Fslf/i8AngscIulN7Zt1eOuC7elk0Nlq4DWSdpLMw/Brkq5o3SAiHouIH6f/vg5YJGlJ1YFGxCPp8y7g0yTzSLTqeTDAsmTFWpM2fRh4OCJuTV9fRXLSbd9m0G2aGWdN2rPV6cDtEfFoh3V1aNM5XeOsSZu+EvhORMxGxJPA1cDL2rbpuT2dDDqIiHdHxJERMUHydfGfImJe5pV0hCSl/15F0pY/qDJOSYdIeubcv4G1wD1tm10DvDntXXAiOQYDLEOeWOvQphHxz8BDko5OF70CmG7bbOBtmifOOrRnm3Ppfull4G3aomucNWnTB4ETJT0jjeUVwEzbNj23Z6ljEw0bzR9k73XA2yXtBfYA50R6G79CzwY+nf5uHghsjojPq56DAeaJtQ5tCrAeuDK9XPBt4C01bdOsOOvSnkh6BrAG+K8ty2rXpjniHHibRsStkq4iuWS1F/gacFm/7enhKMzMzJeJzMzMycDMzHAyMDMznAzMzAwnAzMzw8nARpySUSi7jUq7z/ICjvcbkpa3vP6ipMwJzCU9p4h4JI1L+ny/+7Hh42RgVq3fAJZnbdTBRcDf9HvwiJgFvidpdb/7suHiZGC1llYufy4dGOweSb+dLl8h6UvpoHc3KB2RMf1L+y8l3ZJuvypdvipd9rX0+eiFjtshhsslbU/ff1a6fJ2kqyV9XtI3Jb2/5T3nSfpGGs/fSLpU0stIxrz5cyVj4b8w3fz1km5Lt395lzBeC3w+3fcBkj6gZG6IuyStT5fvlPQ+SV+RNCXp+LRtvjVXkJT6DPDGvJ/fRoMrkK3uTgMeiYgzASQdJmkRySBhZ0XEbJog/gR4a/qeQyLiZUoGwrsceDHwdeCUiNgr6ZXA+0hOsHn8AcmQJG+VdDhwm6R/SNe9FDgO+Blwn6RLgH8D/hfJWEGPA/8E3BkRt0i6hmQ8/KvSzwNwYESsknQG8F6SsWeeIukFwI8i4mfpovNJBik7Lv08v9iy+UMRcZKk/wv8Hck4WwcD9wIb0m2mgD/O+dltRDgZWN3dDXxA0p+RnERvkvRikhP8F9KT6QEkQ/nO2QIQEV+WtDg9gT8T2CTpRSSjNy7qIYa1JAMXvjN9fTDw/PTf/xgRuwEkTQNLgSXAlyLih+nyTwK/ssD+r06fdwATHdY/h2S46jmvBDZExN70c/6wZd016fPdwKER8TjwuKR/lXR4OufBLpLRLs2e4mRgtRYR35C0gmSclT+VtJVkxNN7I+Kkbm/r8PqPgBsj4jclTQBf7CEMAa+NiPvmLZROIPlGMOffSP5PdRo+eCFz+5h7f7s9JAmoNZ5u48jM7evnbbH9vGXfB6f7NHuK7xlYrUl6LvDTiLiCZEKP44H7gHGlc/5KWqT5k4zM3Vc4mWS0xt3AYcB30/XregzjBmC99NRolcdlbH8b8KuSnqVkJqrWy1GPk3xL6cU3mP+NYSvwtnTftF0myuNX2Hd0WxtxTgZWd/+R5Br9HSTX7v84Ip4gGT3yzyTdCdzB/PHcfyTpFpJr5Oely95P8s1iG8llpV78Ecllpbsk3ZO+7iqdiep9wK3AP5AMLb07Xf1x4H+kN6Jf2GUX7fv7CfAtSb+cLvoIyTDGd6Wf/w09fp7/BHyux/fYkPOopTZUJH0ReGdETA04jkMj4sfpX++fBi6PiE/3sb/fBFZExP8sILYvk9x8/1G/+7Lh4W8GZuX43+m3mXuA79Dn9IhpItnZb1CSxoG/cCKwdv5mYGZm/mZgZmZOBmZmhpOBmZnhZGBmZjgZmJkZ8P8BHLYjA7v+aMQAAAAASUVORK5CYII=\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "df.plot(kind='scatter', x='sepal length (cm)', y='petal width (cm)', color='red') \n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "correlation coefficient"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 39,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Correlation:\n"
     ]
    },
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>sepal length (cm)</th>\n",
       "      <th>sepal width (cm)</th>\n",
       "      <th>petal length (cm)</th>\n",
       "      <th>petal width (cm)</th>\n",
       "      <th>Species</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>sepal length (cm)</th>\n",
       "      <td>1.000000</td>\n",
       "      <td>-0.117570</td>\n",
       "      <td>0.871754</td>\n",
       "      <td>0.817941</td>\n",
       "      <td>0.782561</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>sepal width (cm)</th>\n",
       "      <td>-0.117570</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>-0.428440</td>\n",
       "      <td>-0.366126</td>\n",
       "      <td>-0.426658</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>petal length (cm)</th>\n",
       "      <td>0.871754</td>\n",
       "      <td>-0.428440</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>0.962865</td>\n",
       "      <td>0.949035</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>petal width (cm)</th>\n",
       "      <td>0.817941</td>\n",
       "      <td>-0.366126</td>\n",
       "      <td>0.962865</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>0.956547</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Species</th>\n",
       "      <td>0.782561</td>\n",
       "      <td>-0.426658</td>\n",
       "      <td>0.949035</td>\n",
       "      <td>0.956547</td>\n",
       "      <td>1.000000</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                   sepal length (cm)  sepal width (cm)  petal length (cm)  \\\n",
       "sepal length (cm)           1.000000         -0.117570           0.871754   \n",
       "sepal width (cm)           -0.117570          1.000000          -0.428440   \n",
       "petal length (cm)           0.871754         -0.428440           1.000000   \n",
       "petal width (cm)            0.817941         -0.366126           0.962865   \n",
       "Species                     0.782561         -0.426658           0.949035   \n",
       "\n",
       "                   petal width (cm)   Species  \n",
       "sepal length (cm)          0.817941  0.782561  \n",
       "sepal width (cm)          -0.366126 -0.426658  \n",
       "petal length (cm)          0.962865  0.949035  \n",
       "petal width (cm)           1.000000  0.956547  \n",
       "Species                    0.956547  1.000000  "
      ]
     },
     "execution_count": 39,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "print('Correlation:')\n",
    "df.corr()\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 40,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 864x864 with 6 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "import matplotlib.pyplot as plt\n",
    "\n",
    "fig, axes = plt.subplots(3, 2, figsize=(12,12))\n",
    "index = 0\n",
    "for i in range(3):\n",
    "    for j in range(i+1,4):\n",
    "        ax1 = int(index/2)\n",
    "        ax2 = index % 2\n",
    "        axes[ax1][ax2].scatter(df[df.columns[i]], df[df.columns[j]], color='green')\n",
    "        axes[ax1][ax2].set_xlabel(df.columns[i])\n",
    "        axes[ax1][ax2].set_ylabel(df.columns[j])\n",
    "        index = index + 1"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 41,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "-0.11756978413300202"
      ]
     },
     "execution_count": 41,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df['sepal length (cm)'].corr(df['sepal width (cm)'])"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Create a correlation coefficient matrix for 4 features"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 42,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>sepal length (cm)</th>\n",
       "      <th>sepal width (cm)</th>\n",
       "      <th>petal length (cm)</th>\n",
       "      <th>petal width (cm)</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>5.1</td>\n",
       "      <td>3.5</td>\n",
       "      <td>1.4</td>\n",
       "      <td>0.2</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>4.9</td>\n",
       "      <td>3.0</td>\n",
       "      <td>1.4</td>\n",
       "      <td>0.2</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>4.7</td>\n",
       "      <td>3.2</td>\n",
       "      <td>1.3</td>\n",
       "      <td>0.2</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>4.6</td>\n",
       "      <td>3.1</td>\n",
       "      <td>1.5</td>\n",
       "      <td>0.2</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>5.0</td>\n",
       "      <td>3.6</td>\n",
       "      <td>1.4</td>\n",
       "      <td>0.2</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>...</th>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>145</th>\n",
       "      <td>6.7</td>\n",
       "      <td>3.0</td>\n",
       "      <td>5.2</td>\n",
       "      <td>2.3</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>146</th>\n",
       "      <td>6.3</td>\n",
       "      <td>2.5</td>\n",
       "      <td>5.0</td>\n",
       "      <td>1.9</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>147</th>\n",
       "      <td>6.5</td>\n",
       "      <td>3.0</td>\n",
       "      <td>5.2</td>\n",
       "      <td>2.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>148</th>\n",
       "      <td>6.2</td>\n",
       "      <td>3.4</td>\n",
       "      <td>5.4</td>\n",
       "      <td>2.3</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>149</th>\n",
       "      <td>5.9</td>\n",
       "      <td>3.0</td>\n",
       "      <td>5.1</td>\n",
       "      <td>1.8</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>150 rows × 4 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "     sepal length (cm)  sepal width (cm)  petal length (cm)  petal width (cm)\n",
       "0                  5.1               3.5                1.4               0.2\n",
       "1                  4.9               3.0                1.4               0.2\n",
       "2                  4.7               3.2                1.3               0.2\n",
       "3                  4.6               3.1                1.5               0.2\n",
       "4                  5.0               3.6                1.4               0.2\n",
       "..                 ...               ...                ...               ...\n",
       "145                6.7               3.0                5.2               2.3\n",
       "146                6.3               2.5                5.0               1.9\n",
       "147                6.5               3.0                5.2               2.0\n",
       "148                6.2               3.4                5.4               2.3\n",
       "149                5.9               3.0                5.1               1.8\n",
       "\n",
       "[150 rows x 4 columns]"
      ]
     },
     "execution_count": 42,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df_features = df.iloc[:,:4]\n",
    "df_features"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 43,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>sepal length (cm)</th>\n",
       "      <th>sepal width (cm)</th>\n",
       "      <th>petal length (cm)</th>\n",
       "      <th>petal width (cm)</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>sepal length (cm)</th>\n",
       "      <td>1.000000</td>\n",
       "      <td>-0.117570</td>\n",
       "      <td>0.871754</td>\n",
       "      <td>0.817941</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>sepal width (cm)</th>\n",
       "      <td>-0.117570</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>-0.428440</td>\n",
       "      <td>-0.366126</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>petal length (cm)</th>\n",
       "      <td>0.871754</td>\n",
       "      <td>-0.428440</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>0.962865</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>petal width (cm)</th>\n",
       "      <td>0.817941</td>\n",
       "      <td>-0.366126</td>\n",
       "      <td>0.962865</td>\n",
       "      <td>1.000000</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                   sepal length (cm)  sepal width (cm)  petal length (cm)  \\\n",
       "sepal length (cm)           1.000000         -0.117570           0.871754   \n",
       "sepal width (cm)           -0.117570          1.000000          -0.428440   \n",
       "petal length (cm)           0.871754         -0.428440           1.000000   \n",
       "petal width (cm)            0.817941         -0.366126           0.962865   \n",
       "\n",
       "                   petal width (cm)  \n",
       "sepal length (cm)          0.817941  \n",
       "sepal width (cm)          -0.366126  \n",
       "petal length (cm)          0.962865  \n",
       "petal width (cm)           1.000000  "
      ]
     },
     "execution_count": 43,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "corr_mat = df_features.corr()\n",
    "corr_mat"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Make a heat map of the correlation coefficient matrix (1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 44,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 2 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "import seaborn as sns\n",
    "import matplotlib.pyplot as plt\n",
    "sns.heatmap(corr_mat)\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### [Problem 8] Explaining the results of visualized graphs and tables"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "###### THE HEATMAP \n",
    "- It is a square matrix – each row represents a variable, and all the columns represent the same variables as rows, hence the number of rows = number of columns.\n",
    "- Each cell in the grid represents the value of the correlation coefficient between two variables.\n",
    "- A value near to 0 (both positive or negative) indicates the absence of any correlation between the two variables, and hence those variables are independent of each other.\n",
    "- Each cell in the above matrix is also represented by shades of a color. Here darker shades of the color indicate smaller values while brighter shades correspond to larger values (near to 1).\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "##### The boxplot\n",
    "- The boxplot shows the distribution of the data and spot the outliers effectively.\n",
    "- The first thing you might notice in the preceding diagram is a box that contains a horizontal line.\n",
    "\n",
    "- The box represents two inner quartiles where 50% of the data resides, and it ranges from the first quartile to the third quartile.\n",
    "\n",
    "- The horizontal line represents the median of the data."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.5"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
