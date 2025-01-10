import iso2mesh as i2m


#test primitive
n,f,e = i2m.meshabox(np.array([0,0,0]), np.array([1,1,1]), 1)


n,f,e = i2m.meshacylinder(np.array([0,0,0]), np.array([0,0,1]),1)

n,f,e = i2m.meshunitsphere(1)

n,f,e = i2m.meshasphere(np.array([0,0,0]), 5, 2, maxvol=5)
