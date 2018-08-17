from matplotlib.pyplot import figure, scatter, title, legend, show, xlabel, ylabel, contour, contourf
from mpl_toolkits.mplot3d import Axes3D
from numpy import where, meshgrid, arange, vstack, linspace
from numpy.random import randint, seed
from pandas import read_csv

class Tools:
	def __init__(self):
		pass	

	@staticmethod
	def plotData(X, y, p_title="Artificial Dataset", labels=["X_1", "X_2"]):
		n_classes = y.shape[0]
		colorpad = ["#d11141", "#00aedb", "#00b159",  "#ffc425"]

		if(n_classes > 1):
			figure()

			for i in range(0,n_classes):
				class_idx = where(y[i,:] == 1)
				scatter(X[0,class_idx], X[1,class_idx], marker="o", color=colorpad[i], edgecolor="#2A2A2A", label="Class "+str(i+1))

			title(p_title); xlabel(labels[0]); ylabel(labels[1])
			legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
			show()
		else:
			figure()

			for i in range(0,n_classes+1):
				class_idx = where(y == i)
				scatter(X[0,class_idx], X[1,class_idx], marker="o", color=colorpad[i], edgecolor="#2A2A2A", label="Class "+str(i))

			title(p_title); xlabel(labels[0]); ylabel(labels[1])
			legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
			show()


	@staticmethod
	def plotClassContour(X, y, model, p_title="Artificial Dataset", labels=["X_1", "X_2"]):
		n_classes = y.shape[0]
		colorpad = ["#d11141", "#00aedb", "#00b159",  "#ffc425"]
		colorpadBG = ["#d1114160", "#00aedb60", "#00b15960",  "#ffc42560"]

		figure()

		xx, yy = meshgrid(arange(X[0,:].min()-1, X[0,:].max()+1, .1),
						  arange(X[1,:].min()-1, X[1,:].max()+1, .1))
		data = vstack([xx.ravel(), yy.ravel()])
		
		z = model.predict(data)

		zz = z.reshape(xx.shape)

		contour(xx, yy, zz, colors="#2A2A2A", linewidths=.5)
		contourf(xx, yy, zz, levels=range(-1,4), colors=colorpadBG)
		
		if(n_classes > 1):
			for i in range(0,n_classes):
				class_idx = where(y[i,:] == 1)
				scatter(X[0,class_idx], X[1,class_idx], marker="o", color=colorpad[i], edgecolor="#2A2A2A", label="Class "+str(i+1))
		else:
			for i in range(0,n_classes+1):
				class_idx = where(y == i)
				scatter(X[0,class_idx], X[1,class_idx], marker="o", color=colorpad[i], edgecolor="#2A2A2A", label="Class "+str(i))

		title(p_title); xlabel(labels[0]); ylabel(labels[1])
		legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
		show()

	@staticmethod
	def plotClassWireframe(X, y, model, p_title="Artificial Dataset", labels=["X_1", "X_2"]):
		n_classes = y.shape[0]
		colorpad = ["#d11141", "#00aedb", "#00b159",  "#ffc425"]
		colorpadBG = ["#d1114160", "#00aedb60", "#00b15960",  "#ffc42560"]
		
		fig = figure()
		ax = fig.add_subplot(111, projection='3d')

		xx, yy = meshgrid(linspace(X[0,:].min()-10, X[0,:].max()+10, 50), 
						  linspace(X[0,:].min()-10, X[0,:].max()+10, 50))
		data = vstack([xx.ravel(), yy.ravel()])

		z = model.forward(data)[-1]
		
		zz = z.reshape(xx.shape)

		ax.plot_wireframe(xx,yy,zz)

		title(p_title); xlabel(labels[0]); ylabel(labels[1])

		show()

	@staticmethod
	def loadData(path):
		# Carregando os dados a partir do arquivo .csv
		data = read_csv(path)

		# Armazenando as dimensões dos dados
		m = data.shape[0]
		n = data.shape[1]-1

		# Separação do Conjunto de Treino e Conjunto de Teste
		seed(2)
		trainingSize = int(0.8 * m)
		indexes = randint(0, m, m)

		trainData = data.iloc[indexes[:trainingSize]]
		testData = data.iloc[indexes[trainingSize:]]

		# Obtendo matrizes (formato Numpy) correspondentes
		X_train = trainData.iloc[:,:2].values.T
		y_train = trainData.iloc[:,2:].values.T

		X_test = testData.iloc[:,:2].values.T
		y_test = testData.iloc[:,:2].values.T

		return (X_train, X_test, y_train, y_test)