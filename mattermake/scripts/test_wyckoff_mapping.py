from mattermake.data.components.space_group_wyckoff_mapping import (
    SpaceGroupWyckoffMapping,
)


def main():
    print("Testing SpaceGroupWyckoffMapping class...")

    # Test with default path resolution
    sg_map = SpaceGroupWyckoffMapping(
        csv_dir="/p/home/jusers/chandran1/jureca/hai_solaihack/mattermake/mattermake/data"
    )

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
