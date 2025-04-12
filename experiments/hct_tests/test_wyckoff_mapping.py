from mattermake.utils.hct_wyckoff_mapping import (
    SpaceGroupWyckoffMapping,
)


def main():
    print("Testing SpaceGroupWyckoffMapping class...")

    sg_map = SpaceGroupWyckoffMapping()

    for sg in [1, 2, 5, 166, 266]:
        wyckoffs = sg_map.get_allowed_wyckoff_positions(sg)
        print(f"Space group {sg} Wyckoff positions: {wyckoffs}")

        for letter in wyckoffs:
            multiplicity = sg_map.get_wyckoff_multiplicity(sg, letter)
            print(f"  - Position {letter}: multiplicity = {multiplicity}")

    # Test mask creation
    mask = sg_map.create_wyckoff_mask(2)
    print(f"Wyckoff mask for space group 2: {mask}")

    print("Test completed!")


if __name__ == "__main__":
    main()
