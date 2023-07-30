import torch

def inverse_quaternion(quaternion):
    """Oblicza kwaternion odwrotny"""
    w, x, y, z = quaternion
    return torch.tensor([w, -x, -y, -z])

def quaternion_product(q1, q2):
    """Oblicza iloczyn dwóch kwaternionów"""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 + y1 * w2 + z1 * x2 - x1 * z2
    z = w1 * z2 + z1 * w2 + x1 * y2 - y1 * x2
    return torch.tensor([w, x, y, z])

def quaternion_to_euler_angle(q):
    """Konwertuje kwaternion na kąty Eulera"""
    w, x, y, z = q
    ysqr = y * y

    t0 = 2.0 * (w * x + y * z)
    t1 = 1.0 - 2.0 * (x * x + ysqr)
    X = torch.atan2(t0, t1)

    t2 = 2.0 * (w * y - z * x)
    t2 = 1.0 if t2 > 1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    Y = torch.asin(t2)

    t3 = 2.0 * (w * z + x * y)
    t4 = 1.0 - 2.0 * (ysqr + z * z)
    Z = torch.atan2(t3, t4)

    return torch.tensor([X, Y, Z])

def angular_error(q1, q2):
    inverse_q1 = inverse_quaternion(q1)
    angle_error_quat = quaternion_product(inverse_q1, q2)
    angle_error = quaternion_to_euler_angle(angle_error_quat)
    print(angle_error)
