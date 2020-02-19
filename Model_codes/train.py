import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"
import json
import numpy as np
from build_model import build_model_observeLSTM, masked_relative_error
from matplotlib import pyplot as plt
from keras.optimizers import Adam
def smape(A, F):
    return (100/len(A)) * np.sum((2 * np.abs(F - A)) / (np.abs(A) + np.abs(F) + np.finfo(float).eps))
n_input = np.load('../Data_processing/news_text.npy')[:-1]
submission = np.load('../Data_processing/submission_text_october.npy')[:len(n_input)+1]
comment_count = np.load('../Data_processing/temporal_cc10min_october.npy')[:len(n_input)+1]
subred = np.load('../Data_processing/submission_subred_october.npy')[:len(n_input)+1]
comment_rate = np.load('../Data_processing/submission_comment_rate_october.npy')[:len(n_input)]
comment_rate = np.reshape(comment_rate, comment_rate.shape+(1,))
s_input = submission[:-1]
c_count = comment_count[1:]
c_count = c_count.reshape((c_count.shape[0], c_count.shape[1], c_count.shape[2], 1))
s_pred = submission[1:]
subred_input = subred[:-1]
subred_pred = subred[1:]
s_value = np.load('../Data_processing/submission_value_new_october.npy')[1:len(n_input)+1]
_, news_per_hour, token_per_news = n_input.shape
_, sub_per_hour, token_per_sub = s_input.shape
comment_steps = c_count.shape[-2]
#print(news_per_hour, token_per_news, sub_per_hour, sub_per_news)
model = build_model_observeLSTM(news_per_hour,
                    token_per_news,
                    sub_per_hour,
                    token_per_sub,
                                comment_steps)
#model.load_weights('model_observeLSTM_exp-gru.h5')
model.compile(loss=masked_relative_error(0.), optimizer=Adam(clipnorm=1.0))
for i in range(10):
    model.fit([n_input[:30000],
               s_input[:30000],
               s_pred[:30000],
               subred_input[:30000],
               subred_pred[:30000],
               comment_rate[:30000],
               c_count[:30000]],
                    s_value[:30000],
                    #validation_data = ([n_input[-15000:], s_input[-15000:], s_pred[-15000:], subred_input[-15000:], subred_pred[-15000:], ], s_value[-15000:]),
                    batch_size=1,
                    epochs=1)
    if i<9:
        model.reset_states()
    model.save('model_observeLSTM_exp-gru.h5')


'''
pred_value = model.predict([n_input[-15000:], s_input[-15000:], s_pred[-15000:], subred_input[-15000:], subred_pred[-15000:]], batch_size=1, verbose=1)

A = np.reshape(s_value[-15000:], (15000*s_value.shape[1]*s_value.shape[2]))
F = np.reshape(pred_value, (15000*s_value.shape[1]*s_value.shape[2]))

print('sMAP Error:', smape(A, F))
plt.plot(list(range(A.shape[0])), list(A), color='green')
plt.plot(list(range(A.shape[0])), list(F), color='red')
plt.show()
'''
