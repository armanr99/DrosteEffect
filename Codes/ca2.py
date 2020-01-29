import cv2
import numpy as np
from math import e, pi

def getLn(z, r1, r2):
	if(abs(z) > r1 and abs(z) < r2):
		return np.log(z / r1)
	else:
		return 0

def getZLn(z, r1, r2):
	for i in range(0, len(z)):
		for j in range(0, len(z[0])):
			z[i][j] = getLn(z[i][j], r1, r2)
	return z

def getZRotate(z, theta):
	z *= e **(theta * 1j)
	return z

def createImage(img, x, y, repeat = 0):
	if(repeat):
		newImg = np.zeros([len(x[0]), len(x), 3])
	else:
		newImg = np.zeros([len(x), len(x), 3])

	for i in range(0, len(x)):
		for j in range(0, len(x[0])):
			newX = int(x[i][j])
			newY = int(y[i][j])
			if(newX == len(x)):
				newX -= 1
			if(newY == len(y[0])):
				newY -= 1

			newImg[newY][newX][0] = img[i][j % len(x)][0]
			newImg[newY][newX][1] = img[i][j % len(x)][1]
			newImg[newY][newX][2] = img[i][j % len(x)][2]

	return newImg

def convertImage(img, wz, fileName, repeat = 0):
	r = len(img)
	c = len(img[0])

	wx = wz.real
	wy = wz.imag

	wxMax = np.max(abs(wx))
	wyMax = np.max(abs(wy))

	xNew = ((wx / wxMax) + 1) * c / 2
	yNew = ((wy / wyMax) + 1) * r / 2

	newImg = createImage(img, xNew, yNew, repeat)
	cv2.imwrite(fileName, newImg)

def part1(img, z, r1, r2):
	wz = getZLn(z, r1, r2)
	convertImage(img, wz, "part1.jpg")

def part2(img, z, theta):
	wz = getZRotate(z, theta)
	convertImage(img, wz, "part2.jpg")

def part3(img, z, r1, r2, repeat):
	r = len(img)
	c = len(img[0])

	wz = getZLn(z, r1, r2)

	wx = wz.real
	wy = wz.imag

	wxMax = np.max(abs(wx))
	wyMax = np.max(abs(wy))

	xNew = ((wx / wxMax) + 1) * c / 2
	yNew = ((wy / wyMax) + 1) * r / 2

	xNew = np.tile(xNew, repeat)
	yNew = np.tile(yNew, repeat)

	for i in range(1, repeat):
			for k in range(0, c):
				for l in range(i * r, repeat * r):
					yNew[k][l] += r

	newImg = createImage(img, xNew, yNew, 1)
	cv2.imwrite("part3.jpg", newImg)

def part4(img, z, r1, r2, repeat):
	r = len(img)
	c = len(img[0])

	wz = getZLn(z, r1, r2)

	wx = wz.real
	wy = wz.imag

	wxMax = np.max(abs(wx))
	wyMax = np.max(abs(wy))

	xNew = ((wx / wxMax) + 1) * c / 2
	yNew = ((wy / wyMax) + 1) * r / 2

	xNew = np.tile(xNew, repeat)
	yNew = np.tile(yNew, repeat)

	for i in range(1, repeat):
			for k in range(0, c):
				for l in range(i * c, repeat * c):
					yNew[k][l] += c


	xNewMax = np.max(xNew)
	yNewMax = np.max(yNew)
	newZ = ((xNew * 2 / c - 1) * wxMax) + (yNew * 2 / r - 1) * wyMax * 1j

	alpha = np.arctan(np.log(r2 / r1) / (2 * pi))
	f = np.cos(alpha)

	newZ *= (f * e ** (alpha * 1j))
	newZ = e ** newZ;

	convertImage(img, newZ, "part4.jpg")

def main():
	img = cv2.imread("clock.jpg")

	r = len(img)
	c = len(img[0])

	x = np.linspace(-1, 1, r)
	y = np.linspace(-1, 1, c)

	xx, yy = np.meshgrid(x, y)

	z1 = np.vectorize(complex)(xx, yy)
	z2 = np.vectorize(complex)(xx, yy)
	z3 = np.vectorize(complex)(xx, yy)
	z4 = np.vectorize(complex)(xx, yy)

	r1 = 0.4
	r2 = 0.9

	theta = pi / 4

	repeat = 7

	part1(img, z1, r1, r2)
	part2(img, z2, theta)
	part3(img, z3, r1, r2, repeat)
	part4(img, z4, r1, r2, repeat)

if __name__ == "__main__":
	main()
