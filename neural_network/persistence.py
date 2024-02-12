


def save_summary(self, store_large=False):
    '''Write key stats to file, primarily .csv format.
    Weight matrices for `min_loss` and `max_acc` can be saved using
    `store_large`, but file sizes are quite large, even when compressed.'''
    # Unique ID for model
    id = hash(time.time())
    folder = f'trained_models/{id}'
    os.mkdir(folder)

    # Store extended stats in folder for model
    self.epoch_performance.to_csv(f'{folder}/epoch_performance.csv')
    hp_df = pd.DataFrame.from_dict(self.hyperparams, orient='index', columns=['Value'])
    hp_df.index.name = 'Hyperparameter'
    hp_df.to_csv(f'{folder}/hyperparams.csv')
    if store_large:
        self.p_save(self.min_loss_weights, f'{folder}/min_loss_weights')
        self.p_save(self.max_acc_weights,  f'{folder}/max_acc_weights')

    # Append one-liner summary to summary file
    summary_df = pd.DataFrame(self.hyperparams, index=[id])
    summary_df.insert(0, 'min_loss_val', self.min_loss)
    summary_df.insert(1, 'max_acc_val',  self.max_acc)
    summary_df.insert(
        len(summary_df.columns), 'epochs_min_loss_val',
        np.argmin(self.epoch_performance.loss_val)
    )
    summary_df.insert(
        len(summary_df.columns), 'total_sec',
        self.epoch_performance.total_sec_elapsed[self.epoch_index]
    )
    try: # In case `summaries.csv` not yet created
        prev_summaries = pd.read_csv('summaries.csv', index_col=0)
        all_summaries  = self.df_add_dict(prev_summaries, summary_df, id)
    except FileNotFoundError:
        all_summaries = summary_df
    # Write backup of summaries then main summaries file
    all_summaries.to_csv(f'summary_backups/after_{id}.csv')
    all_summaries.to_csv('summaries.csv')