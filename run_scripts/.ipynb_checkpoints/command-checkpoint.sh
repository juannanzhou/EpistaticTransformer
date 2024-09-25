python3 run_script-CLI.py\
--device cuda:0 \
--data_name Faure2023_1_lenient \
--prefix GRB-HD \
--train_percent 10000\
--fit_linear\
--seed 0\
--specify_train\
--train_list ../Data/Data_prepared/train_lists/Faure2023_1_lenient_HD14_train_list_rep_0.pkl\
--iter2 40\
--iter4 40\
--iter8 40


python3 run_script-CLI.py\
--device cuda:0 \
--data_name Faure2023_1_lenient \
--prefix GRB-HD \
--train_percent 10000\
--fit_linear\
--seed 0\
--specify_train\
--train_list ../Data/Data_prepared/train_lists/Faure2023_1_lenient_HD14_train_list_rep_1.pkl\
--iter2 40\
--iter4 40\
--iter8 40


