Collecting workspace informationBelow updates add:
- WeightedRandomSampler for class balancing.
- Softer OneCycleLR (lowered max_lr, gentler warmup).
- Proper temporal validation split, early stopping, and restore-best-weights.

````python
# ...existing code...
def prep_training_data(ml_data, sequence_length=16, test_size=0.2,
                        neutral_band: Tuple[float, float] = (0.40, 0.60)
                    ) -> dict[str, Any]:
    """
    Prepare data for PyTorch training
    Args:
        ml_data (pd.DataFrame): Your dataset
        sequence_length (int): Number of days to look back (for LSTM)
        test_size (int): Fraction for testing
    Returns:
        data (dict): With torch float sensors sent to the {device}
    """
    # Separate features and target
    feature_cols = [col for col in ml_data.columns if col not in ['Target', 'Ticker']]    
    ml_data.dropna(inplace=True)
        
    print(f"📋 Input data shape: {ml_data.shape}")
    
    # Per-ticker temporal split
    X_tr, y_tr, t_tr = [], [], []
    X_va, y_va, t_va = [], [], []  # NEW: validation
    X_te, y_te, t_te = [], [], []

    for tic, g in ml_data.groupby('Ticker', sort=False):
        n = len(g)
        if n < 32:  # skip very short segments
            print('skipped short segment')
            continue
        split = int(n * (1 - test_size))
        gtr = g.iloc[:split] # train+val
        gte = g.iloc[split:] # test

        # temporal validation split from training portion
        val_size = 0.1
        split2 = int(len(gtr) * (1 - val_size))
        gtr_tr = gtr.iloc[:split2]  # train
        gtr_va = gtr.iloc[split2:]  # val

        q_low, q_high = neutral_band
        r = gtr_tr['Target'].to_numpy()
        ql = np.quantile(r, q_low)
        qh = np.quantile(r, q_high) ## LOOK IF THERES ANY COPIES MADE

        def label(ret):
            return 2 if ret > qh else (0 if ret < ql else 1)

        ytr = gtr_tr['Target'].map(label).to_numpy(dtype=np.int64)
        yva = gtr_va['Target'].map(label).to_numpy(dtype=np.int64)
        yte = gte['Target'].map(label).to_numpy(dtype=np.int64)
        
        X_tr.append(gtr_tr[feature_cols].to_numpy(dtype=np.float32, copy=False))
        y_tr.append(ytr)
        t_tr.append(gtr_tr['Ticker'].to_numpy(dtype=np.int16, copy=False))

        X_va.append(gtr_va[feature_cols].to_numpy(dtype=np.float32, copy=False))
        y_va.append(yva)
        t_va.append(gtr_va['Ticker'].to_numpy(dtype=np.int16, copy=False))

        X_te.append(gte[feature_cols].to_numpy(dtype=np.float32, copy=False))
        y_te.append(yte)
        t_te.append(gte['Ticker'].to_numpy(dtype=np.int16, copy=False))
        
    del ml_data
    gc.collect()
    
    # Concatenate splits
    X_train = np.concatenate(X_tr) if X_tr else np.empty((0, len(feature_cols)), dtype=np.float32)
    y_train = np.concatenate(y_tr) if y_tr else np.empty((0,), dtype=np.int64)
    t_train = np.concatenate(t_tr) if t_tr else np.empty((0,), dtype=np.int16)

    X_val = np.concatenate(X_va) if X_va else np.empty((0, len(feature_cols)), dtype=np.float32)
    y_val = np.concatenate(y_va) if y_va else np.empty((0,), dtype=np.int64)
    t_val = np.concatenate(t_va) if t_va else np.empty((0,), dtype=np.int16)

    X_test = np.concatenate(X_te) if X_te else np.empty((0, len(feature_cols)), dtype=np.float32)
    y_test = np.concatenate(y_te) if y_te else np.empty((0,), dtype=np.int64)
    t_test = np.concatenate(t_te) if t_te else np.empty((0,), dtype=np.int16)
    
    # --- FIX: Fit scaler ONLY on training data ---
    feature_scaler = RobustScaler()
    X_train = feature_scaler.fit_transform(X_train).astype(np.float32, copy=False)
    X_val   = feature_scaler.transform(X_val).astype(np.float32, copy=False)
    X_test  = feature_scaler.transform(X_test).astype(np.float32, copy=False)
    
    data = {
        'raw': {
            'X_train': X_train, 'y_train': y_train, 't_train': t_train,
            'X_val':   X_val,   'y_val':   y_val,   't_val':   t_val,
            'X_test':  X_test,  'y_test':  y_test,  't_test':  t_test
        },
        'feature_names': feature_cols,
        'sequence_length': sequence_length
    }
    
    print(f"✅ Regular data - Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    print(f"✅ Sequence data - built lazily in DataLoader")
    
    del X_train, X_val, X_test, y_train, y_val, y_test, t_train, t_val, t_test
    gc.collect()
    
    return data
# ...existing code...
````

````python
# ...existing code...
def train_pytorch_model(model, data, epochs, model_type, batch_size, lr):
    """
    Train PyTorch model and track directional accuracy (Supports GPU)
    """
    import copy
    from torch.utils.data import WeightedRandomSampler

    model = model.to(device).to(dtype=torch.float32)

    # torch.compile can be flaky with cuDNN LSTM; skip for sequence models
    if model_type != 'sequence' and device.type != 'mps':
        try:
            model = torch.compile(model, mode="default", dynamic=True)
        except Exception as e:
            print("compile skipped:", e)
    
    # Build datasets on CPU
    if model_type == 'sequence':
        seq_len = data.get('sequence_length', 16)
        Xtr = data['raw']['X_train']; ytr = data['raw']['y_train']; ttr = data['raw']['t_train']
        Xva = data['raw']['X_val'];   yva = data['raw']['y_val'];   tva = data['raw']['t_val']
        Xte = data['raw']['X_test'];  yte = data['raw']['y_test'];  tte = data['raw']['t_test']

        train_dataset = SequenceDataset(Xtr, ytr, ttr, seq_len)
        val_dataset   = SequenceDataset(Xva, yva, tva, seq_len)
        test_dataset  = SequenceDataset(Xte, yte, tte, seq_len)

        train_labels_np = train_dataset.y[train_dataset.valid_idx].cpu().numpy()
        val_size = len(val_dataset)
    else:
        Xtr = torch.from_numpy(data['raw']['X_train']).to(dtype=torch.float32)
        ytr = torch.from_numpy(data['raw']['y_train'].astype(np.int64, copy=False))
        Xva = torch.from_numpy(data['raw']['X_val']).to(dtype=torch.float32)
        yva = torch.from_numpy(data['raw']['y_val'].astype(np.int64, copy=False))
        Xte = torch.from_numpy(data['raw']['X_test']).to(dtype=torch.float32)
        yte = torch.from_numpy(data['raw']['y_test'].astype(np.int64, copy=False))

        train_dataset = TensorDataset(Xtr, ytr)
        val_dataset   = TensorDataset(Xva, yva)
        test_dataset  = TensorDataset(Xte, yte)

        train_labels_np = ytr.cpu().numpy()
        val_size = len(val_dataset)
        
    # --- Class weights (handle class imbalance) ---
    class_counts = np.bincount(train_labels_np, minlength=3)
    eps = 1e-6
    inv_freq = class_counts.sum() / (class_counts + eps)
    class_weights = inv_freq / inv_freq.mean()
    class_weights_t = torch.tensor(class_weights, dtype=torch.float32, device=device)
    print("train counts: ", class_counts, "percent: " , class_counts/class_counts.sum()*100)

    # --- WeightedRandomSampler (replaces BalancedBatchSampler) ---
    weight_per_class = (1.0 / (class_counts + eps))
    sample_weights = weight_per_class[train_labels_np]
    sampler = WeightedRandomSampler(
        weights=torch.tensor(sample_weights, dtype=torch.double),
        num_samples=len(sample_weights),
        replacement=True
    )
    
    # DataLoaders
    if device.type == 'cuda':
        num_workers = max(1, os.cpu_count() // 2)
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, sampler=sampler,
            pin_memory=True, num_workers=num_workers,
            persistent_workers=(num_workers>0), prefetch_factor=2
        )
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False,
            pin_memory=True, num_workers=num_workers,
            persistent_workers=(num_workers>0), prefetch_factor=2
        )
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False,
            pin_memory=True, num_workers=num_workers,
            persistent_workers=(num_workers>0), prefetch_factor=2
        )

        train_loader = CUDAPrefetchLoader(train_loader, device, prefetch=4)
        val_loader   = CUDAPrefetchLoader(val_loader, device, prefetch=2)
        test_loader  = CUDAPrefetchLoader(test_loader, device, prefetch=2)
    else:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)
        val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False)
        test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False)
        
    loss_fn = nn.CrossEntropyLoss(weight=class_weights_t, label_smoothing=0.0)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    
    print("l_smoothing: 0.0, using WeightedRandomSampler and class weights; neutral band is .45/.55")
    
    # Softer OneCycleLR: lower peak LR, longer warmup, gentler schedule
    onecycle_max_lr_scale = 0.5  # reduce peak to 50% of lr
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=lr * onecycle_max_lr_scale,
        epochs=epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.30,           # longer warmup
        div_factor=50,            # lower initial lr
        final_div_factor=1e3,     # less extreme tail
        anneal_strategy='cos'
    )
    
    # Early stopping
    patience = 6
    min_delta = 1e-4
    best_state = None
    best_val_loss = float('inf')
    epochs_no_improve = 0
    
    train_losses, test_losses, class_accuracy, lr_history = [], [], [], []
    for epoch in range(epochs):
        nb = (device.type == 'cuda')
        # Training
        model.train()
        train_loss = 0.0
        for x, y in train_loader:
            optimizer.zero_grad()
            x = x.to(device, non_blocking=nb)
            y = y.to(device, non_blocking=nb)

            outputs = model(x)
            loss = loss_fn(outputs, y)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            train_loss += float(loss.detach().cpu())

        avg_train_loss = train_loss / len(train_loader)

        # Validation
        model.eval()
        val_loss, correct_predictions, total_predictions = 0.0, 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device, non_blocking=nb)
                y = y.to(device, non_blocking=nb)
                
                outputs = model(x)
                loss = loss_fn(outputs, y)

                val_loss += loss.item() * y.size(0)
                preds = outputs.argmax(1)
                total_predictions += y.size(0)
                correct_predictions += (preds == y).sum().item()

        accuracy = 100.0 * correct_predictions / max(1, total_predictions)
        avg_val_loss = val_loss / max(1, total_predictions)

        train_losses.append(avg_train_loss)
        test_losses.append(avg_val_loss)   # keep key name for existing plots
        class_accuracy.append(accuracy)

        cur_lr = optimizer.param_groups[0]["lr"]
        print(f'Epoch {epoch:3d}: Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}, Val Acc: {accuracy:.1f}%, Lr: {cur_lr:.3e}')
        lr_history.append(cur_lr)

        # Early stopping check
        if avg_val_loss + min_delta < best_val_loss:
            best_val_loss = avg_val_loss
            best_state = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"⏹️ Early stopping at epoch {epoch}. Best Val Loss: {best_val_loss:.6f}")
                break
    
    # Restore best weights
    if best_state is not None:
        model.load_state_dict(best_state)
    
    # Final evaluation on TEST set
    model.eval()
    predictions, actuals = [], []
    prob_down, prob_neutral, prob_up = [], [], []
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device, non_blocking=nb)
            y = y.to(device, non_blocking=nb)

            outputs = model(x)

            probs = torch.softmax(outputs, dim=1)
            preds = outputs.argmax(1)
            
            predictions.extend(preds.cpu().numpy().tolist())
            actuals.extend(y.cpu().numpy().tolist())
            prob_down.extend(probs[:, 0].cpu().numpy().tolist())
            prob_neutral.extend(probs[:, 1].cpu().numpy().tolist())
            prob_up.extend(probs[:, 2].cpu().numpy().tolist())

    predictions_np = np.array(predictions)
    actuals_np = np.array(actuals)
    
    # ✅ DEBUG
    print(f"🔍 Debug - Unique predictions: {np.unique(predictions_np)}")
    print(f"🔍 Debug - Unique actuals: {np.unique(actuals_np)}")
    print(f"🔍 Debug - Predictions shape: {predictions_np.shape}")
    print(f"🔍 Debug - Actuals shape: {actuals_np.shape}")
    
    from sklearn.metrics import classification_report, accuracy_score, f1_score
    
    accuracy_final = accuracy_score(actuals_np, predictions_np) * 100
    f1_weighted = f1_score(actuals_np, predictions_np, average='weighted')
    
    print(f"📊 Final Metrics (TEST) - Accuracy: {accuracy_final:.2f}%, F1-Score: {f1_weighted:.4f}")
    print("\n📋 Classification Report:")
    print(classification_report(actuals_np, predictions_np, labels=[0,1,2], target_names=['Down', 'Neutral', 'Up']))
    
    if USE_FULL_DATASET:
            torch.save(model.state_dict(), f'models_full_dataset/{model.__class__.__name__.lower()}_f.pth')
    else:
        torch.save(model.state_dict(), f'models_limited/{model.__class__.__name__.lower()}_l.pth')
    
    return {
        'model': model,
        'train_losses': train_losses,
        'test_losses': test_losses,          # val loss history
        'class_accuracy': class_accuracy,    # val accuracy history
        'predictions': predictions,
        'actuals': actuals_np.tolist(),
        'probs': { 'down': prob_down, 'neutral': prob_neutral, 'up': prob_up },
        'metrics': {'accuracy': accuracy_final, 'f1_score': f1_weighted},
        'lr_history': lr_history
    }
# ...existing code...
````

Notes:
- Visualization remains compatible; “Validation Loss” uses results['test_losses'].
- Sequence and regular models both use WeightedRandomSampler on training data.
- Temporal validation split prevents leakage and enables early stopping.

Files referenced:
- create_model.ipynb (functions updated: prep_training_data, train_pytorch_model)