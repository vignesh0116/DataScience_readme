# retenmod

This is a python port of the R package [foretell](https://cran.r-project.org/web/packages/foretell/foretell.pdf). This projects customer retention rates, see [*“How to Project Customer Retention” Revisited: The Role of Duration Dependence* (Fader et al., 2018)](https://www.sciencedirect.com/science/article/pii/S1094996818300057) for the original formulation and description of the models. 

To install, just use pip:

    pip install retenmod

Only dependencies are scipy and numpy. For a simple example of use for the BdW model:


```python
import retenmod
surv = [100, 86.9, 74.3, 65.3, 59.3]
res = retenmod.bdw(surv,6)

print('Parameter Estimates')
print(res.params)

print('\nProjections')
print(res.proj)

# Showing the full object
res
```

    Parameter Estimates
    [0.25936956 1.72276277 1.58428486]
    
    Projections
    [100.          86.91461917  74.19279214  65.4782142   59.29285689
      54.66042806  51.03612954  48.10353484  45.66783454  43.60259479
      41.82202357]
    




    churnmod(proj=array([100.        ,  86.91461917,  74.19279214,  65.4782142 ,
            59.29285689,  54.66042806,  51.03612954,  48.10353484,
            45.66783454,  43.60259479,  41.82202357]), negLL=122.27493183193245, params=array([0.25936956, 1.72276277, 1.58428486]))



Here is an example taken from [Koper et al. (2002)](https://www.ojp.gov/sites/g/files/xyckuh241/files/media/document/193428.pdf) on police staff retentions. 

![](https://andrewpwheeler.files.wordpress.com/2021/06/staffretention.png)


```python
import matplotlib.pyplot as plt

large = [100,66,56,52,49,47,44,42]
time = [0,1,2,3,4,5,10,15]

# Only fitting with the first 3 values
train, ext = 4, 15
lrg_bdw = retenmod.bdw(large[0:train],ext - train + 1)

# Showing predicted vs observed
pt = list(range(16))
fig, ax = plt.subplots()
ax.plot(pt[1:], lrg_bdw.proj[1:],label='Predicted', 
        c='k', linewidth=2, zorder=-1)
ax.scatter(time[1:],large[1:],label='Observed', 
           edgecolor='k', c='r', s=50, zorder=1)
ax.axvline(train - 0.5, label='Train', color='grey', 
           linestyle='dashed', linewidth=2, zorder=-2)
ax.set_ylabel('% Retaining Position')
ax.legend(facecolor='white', framealpha=1)
plt.xticks(pt[1:])
plt.show()
```


    
![png](README_sas_files/README_sas_3_0.png)
    

