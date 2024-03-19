def rmse(y, y_pred):
    return sum((y_out - y_act) ** 2 for y_out, y_act in zip(y, y_pred))