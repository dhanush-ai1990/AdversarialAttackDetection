from sklearn.externals import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import scipy
stats=joblib.load('stats_for_evaluate_adversarial.pkl')

print (len(stats))



count ={}
avg_cosine ={}
avg_adv={}
all_cosine =[]
all_adv=[]

for i in range(6):
	count[i]=0
	avg_cosine[i]=[]
	avg_adv[i]=[]

for item in stats:
	data=stats[item]
	adv=data[1:]
	for i in range(6):

		if adv[i][2] == True:
			count[i]+=1
			all_cosine.append(adv[i][1])
			all_adv.append(adv[i][3])
		avg_cosine[i].append(adv[i][1])
		avg_adv[i].append(adv[i][3])


success_count =[count[i] for i in range(6)]
avg_adv=[sum(avg_adv[i])/len(avg_adv[i]) for i in avg_adv]
avg_cosine=[sum(avg_cosine[i])/len(avg_cosine[i]) for i in avg_cosine]
count_temp=[i+1 for i in range(6)]
count_temp=[0,80,160,240,320,391]
corrxy = scipy.stats.pearsonr(all_cosine,all_adv)[0]
print (corrxy)

sns.set_style("whitegrid",{"xtick.major.size": 5})
sns.set(font_scale=1.0)
plt.figure(figsize=(5,5))
sns.set_style("darkgrid",{"xtick.major.size": 5})	

plt.plot(count_temp,avg_adv)
#plt.scatter(all_cosine,all_adv)

plt.gcf().subplots_adjust(bottom=0.15)
sns.plt.title('Avg Adersarial Confidence Vs Relative Position of Adversarial Class').set_fontsize('12')
sns.plt.ylabel('Avg Adersarial Confidence').set_fontsize('12')
sns.plt.xlabel('Position of Adversarial Class w.r.t to True Class').set_fontsize('12')


plt.xticks(rotation=45)
plt.savefig("/Users/Dhanush/Desktop/PositionVsAdvConfidence.png", dpi=300)


