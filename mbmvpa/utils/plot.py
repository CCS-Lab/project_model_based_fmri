

'''
# plotting predictive check sample plot
for pi, pivot in enumerate(np.random.choice(y_pred.shape[0]-self.predchk_plotlen,
                                     self.predchk_plotnum)):
    plot_file = f'repeat_{i:0{len(str(self.n_repeat))}}_predictivecheck_{pi+1}.png'
    plt.figure(figsize=(10, 8))

    y_pred[pivot:pivot+self.predchk_plotlen]
    y_test[pivot:pivot+self.predchk_plotlen]
    
'''