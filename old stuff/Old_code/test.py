

def print_enriched_transaction(transaction, params):
    colnames = []
    
    # add raw features names
    colnames.append("transactionID")
    colnames.append("sourceAccountID")
    colnames.append("targetAccountID")
    colnames.append("timestamp")
    
    # add features names for the graph patterns
    for pattern in ['fan', 'degree', 'scatter-gather', 'temp-cycle', 'lc-cycle']:
        if pattern in params:
            if params[pattern]:
                bins = len(params[pattern +'_bins'])
                if pattern in ['fan', 'degree']:
                    for i in range(bins-1):
                        colnames.append(pattern+"_in_bins_"+str(params[pattern +'_bins'][i])+"-"+str(params[pattern +'_bins'][i+1]))
                    colnames.append(pattern+"_in_bins_"+str(params[pattern +'_bins'][i+1])+"-inf")
                    for i in range(bins-1):
                        colnames.append(pattern+"_out_bins_"+str(params[pattern +'_bins'][i])+"-"+str(params[pattern +'_bins'][i+1]))
                    colnames.append(pattern+"_out_bins_"+str(params[pattern +'_bins'][i+1])+"-inf")
                else:
                    for i in range(bins-1):
                        colnames.append(pattern+"_bins_"+str(params[pattern +'_bins'][i])+"-"+str(params[pattern +'_bins'][i+1]))
                    colnames.append(pattern+"_bins_"+str(params[pattern +'_bins'][i+1])+"-inf")

    vert_feat_names = ["fan","deg","ratio","avg","sum","min","max","median","var","skew","kurtosis"]

    # add features names for the vertex statistics
    for orig in ['source', 'dest']:
        for direction in ['out', 'in']:
            # add fan, deg, and ratio features
            for k in [0, 1, 2]:
                if k in params["vertex_stats_feats"]:
                    feat_name = orig + "_" + vert_feat_names[k] + "_" + direction
                    colnames.append(feat_name)
            for col in params["vertex_stats_cols"]:
                # add avg, sum, min, max, median, var, skew, and kurtosis features
                for k in [3, 4, 5, 6, 7, 8, 9, 10]:
                    if k in params["vertex_stats_feats"]:
                        feat_name = orig + "_" + vert_feat_names[k] + "_col" + str(col) + "_" + direction
                        colnames.append(feat_name)

    df = pd.DataFrame(transaction, columns=colnames)
    return df





    # ----------------------------------------------------------------------

    test123 = data['x']
    test123 = test123[['EdgeID', 'from_id', 'to_id', 'Timestamp']]


    gp = GraphFeaturePreprocessor()
    gp.set_params(params)

    test1234 = test123.iloc[0:10000]
    X_train_enriched = gp.fit_transform(test1234.astype("float64"))

    print("Enriched transactions: ")
    test12345 = print_enriched_transaction(X_train_enriched, gp.get_params())
    test12345.columns
    
    # ----------------------------------------------------------------------




        # save model
    # save_model()

    
    if args.banks:
        for bank in args.banks:
            print(f'Currently training bank {bank}')

            
            bank = 1
            # get indices for the given data
            bank_indices = pdt.get_bank_indices(data_for_indices, bank)

            # filter and process the data for the given bank
            train_data = data_processor(data, bank_indices, single_bank=True)
            print('if')
            

            # train model
            #trainer = trainer_class(model, train_data) if model.__name__ == 'GINe' else trainer_class(model(), train_data)
            #trainer = trainer_class(model, train_data) if model.__name__ == 'GINe' else trainer_class(model(), train_data, params=params, num_rounds=num_rounds)

            trainer = trainer_class(args, train_data)
            trained_models[bank] = trainer.train()

            save_model()

            save_model(trained_models[bank], save_direc + f'\\{model.__name__}_bank_{bank}.{file_type}')


            if model.__name__ == 'GINe':
                save_model(trained_models[bank], save_direc + f'\\{model.__name__}_bank_{bank}.{file_type}')
            else:
                save_model(trained_models[bank].model, save_direc + f'\\{model.__name__}_bank_{bank}.{file_type}', trained_models[bank].scaler)

    else:

        print('else')

        return 0

        # no filtering, just process the data
        train_data = data_processor(data, data_for_indices.index.tolist())

        # train model
        #trainer = trainer_class(model, train_data) if model.__name__ == 'GINe' else trainer_class(model(), train_data)
        trainer = trainer_class(model, train_data, **kwargs) if model.__name__ == 'GINe' else trainer_class(model(),train_data, **kwargs)
        trained_models['all_banks'] = trainer.train()

        if model.__name__ == 'GINe':
            save_model(trained_models['all_banks'], save_direc + f'\\{model.__name__}_all_banks.{file_type}')
        else:
            save_model(trained_models['all_banks'].model, save_direc + f'\\{model.__name__}_all_banks.{file_type}', trained_models['all_banks'].scaler)

    
    
    args.split_perc = configs.split_perc
    # prep to save arguments
    args_dict = {'arguments': vars(args), 'model_configs': tu.get_model_configs(args)}
    folder_path = Path(save_direc)
    file_path = folder_path / 'arguments.json'

    # ensure folder exists and save the file
    folder_path.mkdir(parents=True, exist_ok=True)
    file_path.write_text(json.dumps(args_dict, indent=4))

    return 0 #args, configs, save_direc #trained_models


