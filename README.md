### setup


##### <a href='https://conda.io/miniconda.html'>install miniconda</a>

##### pip install
```
pip install -r requirement.txt
```

##### setup jupyter
- <a href='https://qiita.com/shimaken/items/b411de87b00c051e6697'>setup jupyter</a>

```
$ ipython
In [1]: from IPython.lib import passwd
In [2]: passwd()
In [3]: exit
$ jupyter notebook --generate-config
$ echo '''

c.IPKernelApp.pylab = 'inline'
c.NotebookApp.ip = '*'
c.NotebookApp.open_browser = False
c.NotebookApp.port = 8888
c.NotebookApp.password = 'sha1:XXXXXXX（#先ほど保存したハッシュ値を記載）
'''
```

##### get dataset
<a href='https://www.kaggle.com/c/titanic/data'>Titanic</a>
```
mkdir -p dataset/titanic
cd dataset/titanic
wget https://www.kaggle.com/c/titanic/download/train.csv
wget https://www.kaggle.com/c/titanic/download/test.csv
```

### run
```
jupyter notebook
```
<a href='localhost:8888'>open localhost:8888</a> 
