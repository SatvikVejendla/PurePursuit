import matplotlib.pyplot as plt
fig = plt.figure()

#plot

plt.xlim(0, 10)

plt.ylim(0, 10)


def onclick(event):
    if event.dblclick:
         print(event.button)

connection_id = fig.canvas.mpl_connect('button_press_event', onclick)
plt.show()