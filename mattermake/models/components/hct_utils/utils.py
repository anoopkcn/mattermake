import torch


def bound_lattice_lengths(raw_lengths):
    """Convert raw network outputs to realistic lattice length parameters (a, b, c)

    Args:
        raw_lengths: Raw values from the network

    Returns:
        Bounded lattice length values between 2 and 50 Å
    """

    # Check for NaN/Inf first
    clean_values = torch.nan_to_num(raw_lengths, nan=0.0, posinf=10.0, neginf=-10.0)

    # Clamp to reasonable range before sigmoid to avoid numerical instability
    clean_values = torch.clamp(clean_values, min=-10.0, max=10.0)

    # Apply sigmoid and scale to desired range
    result = 2.0 + 48.0 * torch.sigmoid(clean_values)

    # Final safety check
    result = torch.clamp(result, min=2.0, max=50.0)
    return result


def bound_lattice_angles(raw_angles):
    """Convert raw network outputs to realistic lattice angle parameters (alpha, beta, gamma)

    Args:
        raw_angles: Raw values from the network

    Returns:
        Bounded lattice angle values between 30 and 150 degrees
    """
    # Check for NaN/Inf first
    clean_values = torch.nan_to_num(raw_angles, nan=0.0, posinf=10.0, neginf=-10.0)

    # Clamp to reasonable range before sigmoid to avoid numerical instability
    clean_values = torch.clamp(clean_values, min=-10.0, max=10.0)

    # Apply sigmoid and scale to desired range
    result = 30.0 + 120.0 * torch.sigmoid(clean_values)

    # Final safety check
    result = torch.clamp(result, min=30.0, max=150.0)
    return result


def bound_fractional_coords(raw_coords):
    """Convert raw network outputs to fractional coordinates (x, y, z)

    Args:
        raw_coords: Raw values from the network

    Returns:
        Bounded fractional coordinates between 0 and 1
    """
    # Check for NaN/Inf first
    clean_values = torch.nan_to_num(raw_coords, nan=0.5, posinf=10.0, neginf=-10.0)

    # Clamp to reasonable range before sigmoid to avoid numerical instability
    clean_values = torch.clamp(clean_values, min=-10.0, max=10.0)

    # Apply sigmoid to get values in [0,1] range
    result = torch.sigmoid(clean_values)

    # Ensure coordinates are exactly in [0,1) range for periodic boundary conditions
    result = torch.fmod(result, 1.0)
    return result
