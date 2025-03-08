import pandas as pd

def gradient_descent (m_current, c_current, point_list, learning_rate):
    m_gradient = 0
    c_gradient = 0
    n = len(point_list)

    for i in range(n):
        this_x = point_list.iloc[i].x
        this_y = point_list.iloc[i].y
        
        y_pred = (m_current * this_x) + c_current

        m_gradient += this_x * (this_y - y_pred)
        c_gradient += (this_y - y_pred)

    m_new = m_current + (2/n) * (m_gradient* learning_rate)
    c_new = c_current + (2/n) * (c_gradient *learning_rate)

    return m_new, c_new

