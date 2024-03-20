import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap

fig = plt.figure(figsize = (12,12))
m = Basemap()
m.drawcoastlines(linewidth=1.0, linestyle='solid', color='black')
m.drawcountries(linewidth=1.0, linestyle='solid', color='k')
m.fillcontinents(color='coral',lake_color='aqua')
m.drawmeridians(range(0, 360, 20), color='k', linewidth=1.0, dashes=[4, 4], labels=[0, 0, 1, 1])
plt.title("Longitude lines", fontsize=20, pad=30)
plt.show()