
"""
FinSight AI - Bank Statement Upload Module

Features:
1. Upload CSV and Excel statements
2. Detect bank automatically
3. Map bank-specific columns
4. Clean transactions
5. Validate transactions
6. Preview original and cleaned data
7. Download cleaned CSV
8. Check PostgreSQL duplicates
9. Perform import dry run
10. Confirm and import synthetic transactions
11. Refresh dashboard after successful import

Uses the existing FinSight AI ETL and database modules.

IMPORTANT:
Use synthetic statements only during development.
"""

import hashlib
import os
import tempfile

import pandas as pd
import streamlit as st

from etl.bank_detector import detect_bank
from etl.mapper import map_columns
from etl.cleaner import clean_bank_statement
from etl.validator import validate_transactions

from db.connection import get_connection
from db.import_preview import preview_import
from db.load_transactions import load_transactions


MAX_FILE_SIZE = 10 * 1024 * 1024


# -------------------------------------------------
# HELPER 1: READ CSV OR EXCEL
# -------------------------------------------------

def read_statement(uploaded_file):

    filename = uploaded_file.name.lower()

    uploaded_file.seek(0)

    if filename.endswith(".csv"):

        try:
            return pd.read_csv(uploaded_file)

        except UnicodeDecodeError:
            uploaded_file.seek(0)

            return pd.read_csv(
                uploaded_file,
                encoding="cp1252"
            )

    if filename.endswith((".xlsx", ".xls")):

        return pd.read_excel(uploaded_file)

    raise ValueError(
        "Unsupported file format. Please upload CSV or Excel."
    )


# -------------------------------------------------
# HELPER 2: FILE HASH
# -------------------------------------------------

def get_file_hash(uploaded_file):

    uploaded_file.seek(0)

    content = uploaded_file.read()

    uploaded_file.seek(0)

    return hashlib.sha256(content).hexdigest()


# -------------------------------------------------
# HELPER 3: DATAFRAME PREVIEW
# -------------------------------------------------

def show_dataframe_preview(df, rows=20):

    if df is None or df.empty:

        st.info("No transactions to display.")
        return

    st.dataframe(
        df.head(rows),
        use_container_width=True,
        hide_index=True
    )


# -------------------------------------------------
# HELPER 4: TEMPORARY CLEANED CSV
# -------------------------------------------------

def run_loader_with_temp_file(valid_rows, dry_run=True):
    """
    Pass cleaned data to the existing PostgreSQL loader.

    The temporary file is always removed afterward.
    """

    temporary_path = None

    try:

        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".csv",
            prefix="finsight_",
            delete=False,
            encoding="utf-8",
            newline=""
        ) as temp_file:

            valid_rows.to_csv(
                temp_file,
                index=False
            )

            temporary_path = temp_file.name

        result = load_transactions(
            temporary_path,
            dry_run=dry_run
        )

        return result

    finally:

        if (
            temporary_path is not None
            and os.path.exists(temporary_path)
        ):
            os.remove(temporary_path)


# -------------------------------------------------
# HELPER 5: CHECK DATABASE TARGET
# -------------------------------------------------

def get_database_name():
    """
    Read the current database name safely.
    Never expose database credentials.
    """

    with get_connection() as connection:

        with connection.cursor() as cursor:

            cursor.execute(
                "SELECT current_database()"
            )

            return cursor.fetchone()[0]


# -------------------------------------------------
# HELPER 6: RESET CHECKS ON NEW UPLOAD
# -------------------------------------------------

def reset_checks_if_new_file(file_hash):

    previous_hash = st.session_state.get(
        "finsight_active_upload_hash"
    )

    if previous_hash != file_hash:

        st.session_state[
            "finsight_active_upload_hash"
        ] = file_hash

        st.session_state.pop(
            "finsight_import_preview",
            None
        )

        st.session_state.pop(
            "finsight_preview_hash",
            None
        )

        st.session_state.pop(
            "finsight_dry_run_result",
            None
        )

        st.session_state.pop(
            "finsight_dry_run_hash",
            None
        )


# -------------------------------------------------
# MAIN STREAMLIT FUNCTION
# -------------------------------------------------

def show_statement_upload():

    st.divider()

    st.header("🏦 Upload Bank Statement")

    st.write(
        "Upload a bank statement to detect the bank, "
        "clean transactions, validate data, "
        "check duplicates, and import into PostgreSQL."
    )

    st.warning(
        "Development mode: use synthetic test "
        "bank statements only. Do not upload "
        "real financial information."
    )

    # -------------------------------------------------
    # STEP 1: FILE UPLOAD
    # -------------------------------------------------

    uploaded_file = st.file_uploader(
        "Choose a bank statement",
        type=["csv", "xlsx", "xls"],
        key="finsight_bank_statement"
    )

    if uploaded_file is None:

        st.info(
            "Upload a CSV or Excel bank statement to begin."
        )

        return

    if uploaded_file.size > MAX_FILE_SIZE:

        st.error(
            "File is too large. Maximum allowed size is 10 MB."
        )

        return

    # -------------------------------------------------
    # STEP 2: PROCESS FILE
    # -------------------------------------------------

    try:

        file_hash = get_file_hash(uploaded_file)

        reset_checks_if_new_file(file_hash)

        original_df = read_statement(uploaded_file)

        if original_df.empty:

            st.error("Uploaded statement is empty.")
            return

        detected_bank = detect_bank(original_df)

        mapped_df = map_columns(original_df)

        cleaned_df = clean_bank_statement(mapped_df)

        report = validate_transactions(cleaned_df)

    except Exception as error:

        st.error(
            "Statement processing failed. "
            "Check the uploaded data."
        )

        st.exception(error)

        return

    # -------------------------------------------------
    # STEP 3: BANK DETECTION
    # -------------------------------------------------

    st.subheader("🏦 Bank Detection")

    st.success(
        f"Detected Bank: {detected_bank}"
    )

    # -------------------------------------------------
    # STEP 4: ORIGINAL STATEMENT
    # -------------------------------------------------

    st.subheader("📊 Original Statement Summary")

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Total Rows",
            len(original_df)
        )

    with col2:

        st.metric(
            "Total Columns",
            len(original_df.columns)
        )

    with col3:

        st.metric(
            "Bank",
            str(detected_bank)
        )

    with st.expander("View Original Statement"):

        show_dataframe_preview(original_df)

    # -------------------------------------------------
    # STEP 5: VALIDATION
    # -------------------------------------------------

    st.subheader("✅ Data Validation")

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Total Transactions",
            report["total_rows"]
        )

    with col2:

        st.metric(
            "Valid Transactions",
            report["valid_count"]
        )

    with col3:

        st.metric(
            "Rejected Transactions",
            report["rejected_count"]
        )

    if report["errors"]:

        st.error(
            "Validation errors were detected."
        )

        for error in report["errors"]:
            st.write(error)

    if report["warnings"]:

        st.warning(
            "Validation warnings were detected."
        )

        for warning in report["warnings"]:
            st.write(warning)

    if report["valid"]:

        st.success(
            "All transactions passed critical validation."
        )

    # -------------------------------------------------
    # STEP 6: CLEANED DATA
    # -------------------------------------------------

    st.subheader("📋 Cleaned Transactions")

    valid_rows = report["valid_rows"]

    show_dataframe_preview(
        valid_rows,
        rows=50
    )

    with st.expander(
        "View Row-Level Validation Report"
    ):

        show_dataframe_preview(
            report["row_report"],
            rows=100
        )

    if not report["rejected_rows"].empty:

        with st.expander(
            "View Rejected Transactions"
        ):

            show_dataframe_preview(
                report["rejected_rows"],
                rows=100
            )

    # -------------------------------------------------
    # STEP 7: DOWNLOAD CLEANED CSV
    # -------------------------------------------------

    if not valid_rows.empty:

        cleaned_csv = valid_rows.to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(
            label="⬇️ Download Cleaned Transactions",
            data=cleaned_csv,
            file_name="finsight_cleaned_transactions.csv",
            mime="text/csv",
            key=f"download_cleaned_{file_hash}"
        )

    # -------------------------------------------------
    # STEP 8: IMPORT READINESS
    # -------------------------------------------------

    st.subheader(
        "🗄️ PostgreSQL Import Readiness"
    )

    if not report["valid"] or valid_rows.empty:

        st.error(
            "Import blocked. Fix validation errors first."
        )

        return

    # -------------------------------------------------
    # STEP 9: DATABASE TARGET
    # -------------------------------------------------

    try:

        database_name = get_database_name()

        st.info(
            f"Connected PostgreSQL database: "
            f"{database_name}"
        )

    except Exception as error:

        st.error(
            "Could not verify the PostgreSQL connection."
        )

        st.exception(error)

        return

    # -------------------------------------------------
    # STEP 10: DUPLICATE CHECK
    # -------------------------------------------------

    st.markdown(
        "### 🔍 Check Existing Transactions"
    )

    if st.button(
        "Check Existing Transactions",
        key="check_existing_transactions"
    ):

        # A fresh duplicate check invalidates the
        # previous dry-run approval.
        st.session_state.pop(
            "finsight_dry_run_result",
            None
        )

        st.session_state.pop(
            "finsight_dry_run_hash",
            None
        )

        st.session_state.pop(
            "finsight_import_preview",
            None
        )

        st.session_state.pop(
            "finsight_preview_hash",
            None
        )

        try:

            with st.spinner(
                "Checking PostgreSQL for duplicates..."
            ):

                preview = preview_import(
                    valid_rows
                )

            st.session_state[
                "finsight_import_preview"
            ] = preview

            st.session_state[
                "finsight_preview_hash"
            ] = file_hash

        except Exception as error:

            st.error(
                "Duplicate check failed."
            )

            st.exception(error)

    preview = None

    if (
        st.session_state.get(
            "finsight_preview_hash"
        ) == file_hash
    ):

        preview = st.session_state.get(
            "finsight_import_preview"
        )

    if preview is not None:

        st.success(
            "PostgreSQL duplicate check completed!"
        )

        col1, col2, col3 = st.columns(3)

        with col1:

            st.metric(
                "Total Transactions",
                preview["total"]
            )

        with col2:

            st.metric(
                "New Transactions",
                preview["new_count"]
            )

        with col3:

            st.metric(
                "Possible Duplicates",
                preview["possible_duplicate_count"]
            )

        if preview["possible_duplicate_count"] > 0:

            st.warning(
                "Possible duplicates detected. "
                "Import will be blocked."
            )

            with st.expander(
                "View Possible Duplicates"
            ):

                show_dataframe_preview(
                    preview["possible_duplicates"],
                    rows=100
                )

        if preview["new_count"] > 0:

            with st.expander(
                "View New Transactions"
            ):

                show_dataframe_preview(
                    preview["new_rows"],
                    rows=100
                )

    # -------------------------------------------------
    # STEP 11: DRY RUN
    # -------------------------------------------------

    st.markdown(
        "### 🧪 Check Import Readiness (Dry Run)"
    )

    st.caption(
        "The dry run validates the prepared data "
        "without writing to PostgreSQL."
    )

    if st.button(
        "Check Import Readiness (Dry Run)",
        key="check_import_dry_run"
    ):

        st.session_state.pop(
            "finsight_dry_run_result",
            None
        )

        st.session_state.pop(
            "finsight_dry_run_hash",
            None
        )

        if preview is None:

            st.warning(
                "Run Check Existing Transactions first."
            )

        elif preview["possible_duplicate_count"] > 0:

            st.error(
                "Dry-run approval blocked because "
                "possible duplicates exist."
            )

        elif preview["new_count"] != len(valid_rows):

            st.error(
                "Not all transactions are confirmed as new."
            )

        else:

            try:

                with st.spinner(
                    "Checking import readiness..."
                ):

                    dry_run_result = run_loader_with_temp_file(
                        valid_rows,
                        dry_run=True
                    )

                if (
                    dry_run_result.get("status") == "dry_run"
                    and dry_run_result.get("ready")
                    == len(valid_rows)
                ):

                    st.session_state[
                        "finsight_dry_run_result"
                    ] = dry_run_result

                    st.session_state[
                        "finsight_dry_run_hash"
                    ] = file_hash

                    st.success(
                        "Dry run passed. "
                        "No transactions were inserted."
                    )

                else:

                    st.error(
                        "Dry run did not pass."
                    )

            except Exception as error:

                st.error(
                    "Import readiness check failed."
                )

                st.exception(error)

    dry_run_result = None

    if (
        st.session_state.get(
            "finsight_dry_run_hash"
        ) == file_hash
    ):

        dry_run_result = st.session_state.get(
            "finsight_dry_run_result"
        )

    if dry_run_result is not None:

        st.success(
            f"Dry run ready: "
            f"{dry_run_result.get('ready', 0)} transactions."
        )

    # -------------------------------------------------
    # STEP 12: IMPORT CONFIRMATION
    # -------------------------------------------------

    st.divider()

    st.subheader(
        "🚀 Confirm Import to PostgreSQL"
    )

    st.write(
        "Import is enabled only after successful "
        "validation, duplicate checking, and dry run."
    )

    # The UI initially permits writes only in the
    # isolated synthetic-data test database.
    # This prevents accidentally importing into
    # the existing main database during testing.

    safe_database = (
        database_name == "finsight_ai_test"
    )

    if not safe_database:

        st.warning(
            "Import is currently restricted to "
            "'finsight_ai_test'. Your main database "
            "will remain unchanged."
        )

    checks_passed = (
        safe_database
        and report["valid"]
        and preview is not None
        and preview["possible_duplicate_count"] == 0
        and preview["new_count"] == len(valid_rows)
        and dry_run_result is not None
        and dry_run_result.get("status") == "dry_run"
        and dry_run_result.get("ready") == len(valid_rows)
    )

    confirmed = st.checkbox(
        "I confirm this is synthetic test data "
        "and I want to import it into PostgreSQL.",
        value=False,
        key=f"confirm_import_{file_hash}"
    )

    import_clicked = st.button(
        "🚀 Import to PostgreSQL",
        type="primary",
        disabled=not (checks_passed and confirmed),
        key="confirm_postgresql_import"
    )

    if import_clicked:

        try:

            # Repeat the duplicate check immediately
            # before import. The database loader also
            # performs its own checks inside a transaction.
            fresh_preview = preview_import(
                valid_rows
            )

            if (
                fresh_preview["possible_duplicate_count"] > 0
                or fresh_preview["new_count"] != len(valid_rows)
            ):

                st.error(
                    "Import cancelled because the database "
                    "contents changed or duplicates were found."
                )

                return

            # Final check: never import if the active
            # configured database is not the test database.
            if get_database_name() != "finsight_ai_test":

                st.error(
                    "Database target changed. Import cancelled."
                )

                return

            with st.spinner(
                "Importing synthetic transactions..."
            ):

                result = run_loader_with_temp_file(
                    valid_rows,
                    dry_run=False
                )

            if result.get("status") == "imported":

                st.success(
                    f"Successfully imported "
                    f"{result['inserted']} transactions!"
                )

                # Clear approvals so another import
                # requires a fresh duplicate check.
                st.session_state.pop(
                    "finsight_import_preview",
                    None
                )

                st.session_state.pop(
                    "finsight_preview_hash",
                    None
                )

                st.session_state.pop(
                    "finsight_dry_run_result",
                    None
                )

                st.session_state.pop(
                    "finsight_dry_run_hash",
                    None
                )

                st.info(
                    "Reload the dashboard to view "
                    "the imported transactions."
                )

            elif result.get("status") == "duplicate_file":

                st.warning(
                    "This statement has already been imported."
                )

            elif result.get("status") == "possible_duplicates":

                st.warning(
                    "Import blocked because possible "
                    "duplicate transactions were detected."
                )

                st.write(
                    result.get(
                        "possible_duplicates",
                        []
                    )
                )

            else:

                st.error(
                    "Import was not completed."
                )

                st.write(
                    "Status:",
                    result.get("status")
                )

        except Exception as error:

            st.error(
                "PostgreSQL import failed."
            )

            st.exception(error)

    st.caption(
        "FinSight AI development environment. "
        "Use synthetic statements only."
    )
