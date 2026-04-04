from panda_matching.cli import main


def test_cli_runs(capsys) -> None:  # type: ignore[no-untyped-def]
    main()
    captured = capsys.readouterr()
    assert "panda-matching import" in captured.out
